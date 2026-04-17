#!/bin/bash
# Direct parallel launcher for ObjectNav to StreamVLN conversion
# Launches 16 parallel tasks across 4 GPUs (4 tasks per GPU)

set -e

# Configuration - 4 GPUs, 4 scenes
GPU_IDS=(3 4 5 6)

SCENES=(
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shanghai-zhujiajiao-room2-1-2025-07-15_14-52-28:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shanghai-zhujiajiao-room3-2025-07-15_15-09-38:data/trajectory_data/hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shenzhen-room_ziwei_20250724-metacam:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/suzhou-room-zhangbo-metacam-2025-07-09_22-27-19:data/trajectory_data/hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
)

echo "========================================="
echo "Batch Parallel Converter"
echo "========================================="
echo "GPUs: ${GPU_IDS[@]}"
echo "Scenes: ${#SCENES[@]}"
echo "Total jobs: $((${#SCENES[@]} * 4) (4 scenes × 4 parts each)"
echo ""

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/conversion_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs: $LOG_DIR"
echo ""

# Function to launch a single job
launch_job() {
    local gpu_id=$1
    local annot_dir=$2
    local data_dir=$3
    local part_num=$4
    local log_file=$5

    local annot_file="${annot_dir}/annotation_${part_num}.json"
    local scene_name=$(basename "$annot_dir")

    # Find matching data file
    local data_file=$(find "$data_dir" -name "${scene_name}.json.gz" 2>/dev/null | head -1)

    if [ ! -f "$annot_file" ]; then
        echo "[Error] File not found: $annot_file"
        return 1
    fi

    if [ -z "$data_file" ]; then
        echo "[Error] Data file not found for scene: $scene_name"
        return 1
    fi

    local job_name="${scene_name}_part${part_num}_gpu${gpu_id}"

    echo "[Starting] $job_name"
    echo "  GPU: $gpu_id"
    echo "  Annotation: annotation_${part_num}.json"
    echo "  Data: $data_file"
    echo "  Log: $log_file"
    echo ""

    # Run in background
    (
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python ./scripts/objnav_converters/objnav2streamvln.py \
            --annot-path "$annot_file" \
            habitat.dataset.data_path="$data_file" \
            > "$log_file" 2>&1

        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[Completed] $job_name"
        else
            echo "[Failed] $job_name (exit code: $exit_code)"
        fi
    ) &

    echo $!
}

# Export function
export -f launch_job

# Track all PIDs
declare -a PIDS

echo "Launching $((${#SCENES[@]} * 4) parallel jobs..."
echo "========================================="
echo ""

# Launch all jobs
job_count=0
for scene_idx in "${!SCENES[@]}"; do
    gpu_idx=$((scene_idx % 4))  # Distribute scenes across 4 GPUs
    gpu_id=${GPU_IDS[$gpu_idx]}

    annot_dir=$(echo "$scene_idx" | cut -d: -f1)
    data_dir=$(echo "$scene_idx" | cut -d: -f2)

    # Launch 4 jobs for this scene
    for part in {0..3}; do
        log_file="${LOG_DIR}/$(basename "$annot_dir")_part${part}_gpu${gpu_id}.log"

        pid=$(launch_job "$gpu_id" "$annot_dir" "$data_dir" "$part" "$log_file")
        PIDS+=($pid)
        job_count=$((job_count + 1))
    done
done

echo ""
echo "========================================="
echo "All $job_count jobs launched in parallel!"
echo "========================================="
echo "Total PIDs: ${#PIDS[@]}"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all jobs
failed_count=0
completed_count=0

for i in "${!PIDS[@]}"; do
    if wait $i; then
        completed_count=$((completed_count + 1))
        echo "[Progress] Completed: $completed_count / $job_count"
    else
        failed_count=$((failed_count + 1))
        echo "[Error] Job failed with PID: $i"
    fi
done

echo ""
echo "========================================="
echo "All jobs completed!"
echo "========================================="
echo "Completed: $completed_count / $job_count"
echo "Failed: $failed_count"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

if [ $failed_count -eq 0 ]; then
    echo "✓ All jobs completed successfully!"
    exit 0
else
    echo "✗ $failed_count jobs failed. Check logs in $LOG_DIR"
    exit 1
fi
