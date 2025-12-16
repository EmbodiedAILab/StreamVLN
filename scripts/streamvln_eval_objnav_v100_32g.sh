# export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export HF_HUB_OFFLINE=0

CHECKPOINT="checkpoints/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
echo "CHECKPOINT: ${CHECKPOINT}"

CONFIG_PATH="config/objnav_hm3d.yaml"

torchrun --nproc_per_node=2 \
        --standalone streamvln/objnav_eval.py \
        --model_path $CHECKPOINT \
        --habitat_config_path $CONFIG_PATH \
        --output_path "results/vals/test/objnav_hm3d"
        # --save_video