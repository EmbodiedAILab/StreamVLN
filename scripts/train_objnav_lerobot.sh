#!/bin/bash
# Training script for ObjectNav LeRobot dataset

export HF_HUB_OFFLINE=1
export HF_HOME=$PWD/checkpoints/hf_home/

# Distributed training parameters
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12000}

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

# Auto-select a free port
if command -v python3 >/dev/null 2>&1; then
SEL_PORT=$(python3 - <<'PY'
import os, socket, random, sys

def is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False

env_port = os.environ.get("MASTER_PORT")
if env_port is not None:
    try:
        p = int(env_port)
        if 1024 <= p <= 65535 and is_free(p):
            print(p)
            sys.exit(0)
    except Exception:
        pass

start = int(os.environ.get("PORT_START", 29500))
end = int(os.environ.get("PORT_END", 65535))
for _ in range(2048):
    p = random.randint(start, end)
    if is_free(p):
        print(p)
        sys.exit(0)

print(0)
PY
)
    if [ -n "$SEL_PORT" ] && [ "$SEL_PORT" != "0" ]; then
        MASTER_PORT=$SEL_PORT
    fi
fi

echo "=== Distributed training config ==="
echo "Num nodes: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Master endpoint: $MASTER_ADDR:$MASTER_PORT"
echo "HF cache dir: $HF_HOME"

# ===== ObjectNav LeRobot Configuration =====
OBJNAV_LEROBOT_ROOT="data/trajectory_data/objectnav/hm3d_v2_lerobot3_test"
USE_OBJNAV_LEROBOT=True

# ===== Model Configuration =====
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="$(printf '%s' "$LLM_VERSION" | tr '/' '_')"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="$(printf '%s' "$VISION_MODEL_VERSION" | tr '/' '_')"

# ===== Run Configuration =====
PROMPT_VERSION="qwen_1_5"
RUN_NAME="StreamVLN_ObjectNav_LeRobot_${PROMPT_VERSION}"
PREV_STAGE_CHECKPOINT="checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2"

echo "=== Run Configuration ==="
echo "OBJNAV_LEROBOT_ROOT: ${OBJNAV_LEROBOT_ROOT}"
echo "USE_OBJNAV_LEROBOT: ${USE_OBJNAV_LEROBOT}"
echo "RUN_NAME: ${RUN_NAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "========================"

# Detect GPU count
GPU_COUNT=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _CVD_CLEAN=$(echo "$CUDA_VISIBLE_DEVICES" | tr -d '[:space:]' | sed 's/,,*/,/g;s/^,//;s/,$//')
    if [ -n "$_CVD_CLEAN" ]; then
        GPU_COUNT=$(echo "$_CVD_CLEAN" | awk -F, '{print NF}')
    else
        GPU_COUNT=0
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]')
fi

if [ -z "$GPU_COUNT" ] || ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    if command -v python3 >/dev/null 2>&1; then
        GPU_COUNT=$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
)
        GPU_COUNT=$(echo "$GPU_COUNT" | tr -d '[:space:]')
    fi
fi

if ! echo "$GPU_COUNT" | grep -Eq '^[0-9]+$'; then
    echo "Error: Unable to determine GPU count."
    exit 1
fi

if [ "$GPU_COUNT" -le 0 ]; then
    echo "Error: No GPUs visible."
    exit 1
fi

echo "Detected GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt "$NPROC_PER_NODE" ]; then
    echo "Adjusting NPROC_PER_NODE to $GPU_COUNT"
    NPROC_PER_NODE=$GPU_COUNT
fi
echo "========================"

# ===== Training =====
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=12345 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    streamvln/streamvln_train.py \
    --deepspeed scripts/zero2_v100_32g.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --use_objnav_lerobot ${USE_OBJNAV_LEROBOT} \
    --objnav_lerobot_root ${OBJNAV_LEROBOT_ROOT} \
    --group_by_task False \
    \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_mlp_adapter,mm_lora_layer" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --fp16 True \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --run_name $RUN_NAME \
    --output_dir checkpoints/$RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --mm_projector_lr 1e-5 \
    --mm_vision_tower_lr 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.075 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 9e-06}' \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --torch_compile False \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --report_to tensorboard

echo "Training completed!"
