#!/bin/bash
# Batch converter for multiple ObjectNav scenes
# This script processes 4 scenes in parallel, each using 4-way split annotations

set -e

# Configuration
GPU_DEVICES=(3 4 5 6)  # 4 GPUs available

# Scenes to process (format: "annot_dir:data_dir")
SCENES=(
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shanghai-zhujiajiao-room2-1-2025-07-15_14-52-28:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shanghai-zhujiajiao-room3-2025-07-15_15-09-38:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/shenzhen-room_ziwei_20250724-metacam:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
    "data/trajectory_data/objectnav/cloudrobo_v1_l3mvn/suzhou-room-zhangbo-metacam-2025-07-09_22-27-19:data/trajectory_data_hm3d_format/objectnav/cloudrobo_v1_l3mvn/train/content"
)

echo "========================================="
echo "Batch ObjectNav Converter"
echo "========================================="
echo "GPUs: ${GPU_DEVICES[@]}"
echo "Scenes: ${#SCENES[@]}"
echo ""

# Create logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/conversion_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Function to process one scene
process_scene() {
    local gpu_id=$1
    local scene_spec=$2
    local log_file=$3

    local annot_dir=$(echo "$scene_spec" | cut -d: -f1)
    local data_dir=$(echo "$scene_spec" | cut -d: -f2)

    echo "[GPU $gpu_id] Processing: $(basename $annot_dir)"

    # Run the parallel conversion script
    ./scripts/objnav_converters/run_parallel_conversion.sh \
        "$gpu_id" \
        "$annot_dir" \
        "$data_dir" > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu_id] ✓ Completed: $(basename $annot_dir)"
    else
        echo "[GPU $gpu_id] ✗ Failed: $(basename $annot_dir) (exit code: $exit_code)"
    fi

    return $exit_code
}

# Export function
export -f process_scene

# Check if we have enough GPUs
if [ ${#SCENES[@]} -gt ${#GPU_DEVICES[@]} ]; then
    echo "Warning: More scenes (${#SCENES[@]}) than GPUs (${#GPU_DEVICES[@]})"
    echo "Only processing first ${#GPU_DEVICES[@]} scenes in parallel"
fi

# Launch jobs
pids=()
log_files=()

for i in "${!GPU_DEVICES[@]}"; do
    if [ $i -lt ${#SCENES[@]} ]; then
        gpu_id=${GPU_DEVICES[$i]}
        scene_spec=${SCENES[$i]}
        log_file="$LOG_DIR/gpu_${gpu_id}_$(echo ${SCENES[$i]} | cut -d/ -f7).log"

        echo "Launching job for GPU $gpu_id..."
        process_scene "$gpu_id" "$scene_spec" "$log_file" &
        pids+=($!)
        log_files+=("$log_file")
    fi
done

echo ""
echo "Waiting for all jobs to complete..."
echo "========================================="
echo ""

# Wait for all jobs and collect results
failed_scenes=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    wait $pid
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        failed_scenes=$((failed_scenes + 1))
    fi
done

# Summary
echo "========================================="
echo "Batch Conversion Complete!"
echo "========================================="
echo "Total scenes: ${#GPU_DEVICES[@]}"
echo "Successful: $((${#GPU_DEVICES[@]} - failed_scenes))"
echo "Failed: $failed_scenes"
echo ""
echo "Logs saved to: $LOG_DIR"
echo ""

if [ $failed_scenes -eq 0 ]; then
    echo "✓ All scenes processed successfully!"
    exit 0
else
    echo "✗ Some scenes failed. Check logs in $LOG_DIR"
    exit 1
fi
