#!/bin/bash
# Quick test training script for ObjectNav LeRobot dataset

export HF_HUB_OFFLINE=1
export HF_HOME=$PWD/checkpoints/hf_home/

# Use single GPU for testing
export NNODES=1
export NPROC_PER_NODE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12000

echo "=========================================="
echo "ObjectNav LeRobot Training - Quick Test"
echo "=========================================="
echo "Using test dataset: data/trajectory_data/objectnav/hm3d_v2_lerobot3_test"
echo "=========================================="

# ===== ObjectNav LeRobot Configuration =====
# Use test dataset
OBJNAV_LEROBOT_ROOT="data/trajectory_data/objectnav/hm3d_v2_lerobot3_test"
USE_OBJNAV_LEROBOT=True

# ===== Model Configuration =====
PROMPT_VERSION="qwen_1_5"
RUN_NAME="StreamVLN_ObjectNav_LeRobot_Test"
PREV_STAGE_CHECKPOINT="checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2"

# ===== Quick Test Configuration =====
NUM_TRAIN_EPOCHS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION=2
LOGGING_STEPS=5
SAVE_STEPS=100

echo "Configuration:"
echo "  Dataset: ${OBJNAV_LEROBOT_ROOT}"
echo "  Epochs: ${NUM_TRAIN_EPOCHS}"
echo "  Batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION}"
echo "=========================================="

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
    --num_frames 16 \
    --data_augmentation True \
    \
    --mm_tunable_parts="mm_mlp_adapter,mm_lora_layer" \
    --vision_tower "google/siglip-so400m-patch14-384" \
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
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --mm_projector_lr 1e-5 \
    --mm_vision_tower_lr 1e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 9e-06}' \
    --logging_steps $LOGGING_STEPS \
    --tf32 False \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --torch_compile False \
    --dataloader_drop_last True \
    --report_to tensorboard \
    --max_steps 20  # Limit to 20 steps for quick test

echo ""
echo "=========================================="
echo "Test completed!"
echo "Check logs above for any errors"
echo "=========================================="
