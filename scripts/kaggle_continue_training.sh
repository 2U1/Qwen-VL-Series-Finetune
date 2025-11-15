#!/bin/bash

# ===== Continue Training from Previous Kaggle Session =====
# Use this to continue training from a saved checkpoint
# =========================================================

# IMPORTANT: Set these variables before running
CHECKPOINT_DATASET="/kaggle/input/zac-checkpoint-300"  # Change this to your checkpoint dataset
CURRENT_STEPS=300  # Steps completed in previous session
TARGET_STEPS=600   # Total steps you want (current + new)

echo "==========================================="
echo "Continuing Training from Previous Session"
echo "==========================================="
echo "Previous checkpoint: $CHECKPOINT_DATASET"
echo "Completed steps: $CURRENT_STEPS"
echo "Target steps: $TARGET_STEPS"
echo "Will train: $((TARGET_STEPS - CURRENT_STEPS)) more steps"
echo "==========================================="

# Set PYTHONPATH
export PYTHONPATH=/kaggle/working/Qwen-VL-Series-Finetune:$PYTHONPATH

# Change to repo directory
cd /kaggle/working/Qwen-VL-Series-Finetune

# Create checkpoint directory if not exists
mkdir -p /kaggle/working/checkpoints/zac_qwen2vl_lora

# Copy checkpoint from input dataset
echo "Copying checkpoint from dataset..."
if [ -d "$CHECKPOINT_DATASET" ]; then
    cp -r ${CHECKPOINT_DATASET}/checkpoint-* /kaggle/working/checkpoints/zac_qwen2vl_lora/ 2>/dev/null || {
        # Try without nested checkpoint- folder
        cp -r ${CHECKPOINT_DATASET}/* /kaggle/working/checkpoints/zac_qwen2vl_lora/
    }
    echo "✅ Checkpoint copied successfully"
    ls -lh /kaggle/working/checkpoints/zac_qwen2vl_lora/
else
    echo "❌ Checkpoint dataset not found: $CHECKPOINT_DATASET"
    echo "Please check your Kaggle notebook inputs"
    exit 1
fi

# Model and training config
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=2
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "==========================================="
echo "Starting Training..."
echo "==========================================="

# Run training with max_steps
deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id "$MODEL_NAME" \
    --data_path /kaggle/input/zac-sample-600/zac_llava_format.json \
    --image_folder /kaggle/working/videos \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --lora_enable True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --bf16 False \
    --fp16 True \
    --disable_flash_attn2 False \
    --output_dir /kaggle/working/checkpoints/zac_qwen2vl_lora \
    --max_steps $TARGET_STEPS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((10 * 14 * 4 * 28 * 28)) \
    --video_max_pixels $((16 * 14 * 4 * 28 * 28 - 10000)) \
    --fps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir /kaggle/working/logs \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --save_latest_only True \
    --dataloader_num_workers 1

echo "==========================================="
echo "✅ Training session completed!"
echo "==========================================="

# Save checkpoint for next session
bash scripts/kaggle_resume_helper.sh save

echo "==========================================="
echo "📌 Next steps:"
echo "1. Check /kaggle/working/outputs for new checkpoint"
echo "2. Upload it to Kaggle Datasets"
echo "3. Update CHECKPOINT_DATASET and TARGET_STEPS in this script"
echo "4. Run again for next session"
echo "==========================================="
