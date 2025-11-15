#!/bin/bash

# ===== ZAC2025 Traffic Sign Recognition Training =====
# Model: Qwen2-VL-2B with LoRA
# =====================================================

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# Set the PYTHONPATH to include the necessary source directory
export PYTHONPATH=/content/Qwen-VL-Series-Finetune:$PYTHONPATH

# ===== Training Config =====
GLOBAL_BATCH_SIZE=8  # Effective batch size
BATCH_PER_DEVICE=1   # Per GPU batch size  
NUM_DEVICES=1        # Single T4 GPU
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))  # Gradient accumulation steps

echo "==========================================="
echo "ZAC2025 Training Configuration"
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
    --model_id $MODEL_NAME \
    --data_path /content/zac_llava_format.json \
    --image_folder /content/traffic_buddy_train+public_test \
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
    --disable_flash_attn2 True \
    --output_dir /content/checkpoints/zac_qwen2vl_lora_full \
    --num_train_epochs 10 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((2 * 28 * 28)) \
    --video_max_pixels $((2 * 28 * 28)) \
    --fps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir ./logs \
    --lazy_preprocess False \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --dataloader_num_workers 2

echo "==========================================="
echo "✅ Training completed!"
echo "==========================================="
