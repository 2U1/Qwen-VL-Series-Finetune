#!/bin/bash

# ===== ZAC2025 Training Configuration for Kaggle =====
# Model: Qwen2-VL-2B with LoRA
# ======================================================

# CRITICAL: Set PYTHONPATH for Kaggle environment
export PYTHONPATH=/kaggle/working/Qwen-VL-Series-Finetune:$PYTHONPATH

# Change to the repo directory
cd /kaggle/working/Qwen-VL-Series-Finetune

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# ===== Training Config =====
GLOBAL_BATCH_SIZE=4  # Effective batch size
BATCH_PER_DEVICE=1   # Per GPU batch size
NUM_DEVICES=1        # Single GPU on Kaggle
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

echo "==========================================="
echo "ZAC2025 Training Configuration (Kaggle)"
echo "==========================================="
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation: $GRAD_ACCUM_STEPS"
echo "==========================================="

# Start the training with DeepSpeed and the configured parameters
deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id "$MODEL_NAME" \
    --data_path /kaggle/input/zac-sample-600/zac_llava_format.json \
    --image_folder /kaggle/input/train-zaic/videos \
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
    --num_train_epochs 5 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((14 * 14 * 4 * 28 * 28)) \
    --video_max_pixels $((16 * 14 * 4 * 28 * 28)) \
    --fps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir /kaggle/working/logs \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --dataloader_num_workers 1

echo "==========================================="
echo "✅ Training completed!"
echo "Checkpoints saved to: /kaggle/working/checkpoints/zac_qwen2vl_lora"
echo "==========================================="
