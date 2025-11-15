#!/bin/bash

# ===== ZAC2025 Training Phase 2a - Incremental Unfreezing =====
# Resume từ Phase 1 và unfreeze TOP 6 layers của LLM
# Tiết kiệm memory hơn so với unfreeze toàn bộ LLM
# =============================================================

export PYTHONPATH=/kaggle/working/Qwen-VL-Series-Finetune:$PYTHONPATH
cd /kaggle/working/Qwen-VL-Series-Finetune

MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# ===== Configuration =====
# Giảm batch size và resolution để tránh OOM
GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1   # Giảm xuống 1 (thay vì 2)
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# Checkpoint từ phase 1
RESUME_FROM="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"

# Output directory mới cho phase 2a
OUTPUT_DIR="/kaggle/working/checkpoints/zac_qwen2vl_phase2a"

echo "==========================================="
echo "ZAC2025 Training Phase 2a (Unfreeze Top-6 LLM Layers)"
echo "==========================================="
echo "Resume from: $RESUME_FROM"
echo "Output to: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "Batch per Device: $BATCH_PER_DEVICE"
echo "Gradient Accumulation: $GRAD_ACCUM_STEPS"
echo "Unfreeze: Top 6 layers of LLM (out of 28 layers)"
echo "==========================================="

# Copy checkpoint to new output directory for resume
echo "Preparing checkpoint for resume..."
mkdir -p "$OUTPUT_DIR"
cp -r "$RESUME_FROM"/* "$OUTPUT_DIR/"
echo "✅ Checkpoint copied"

# Start training with TOP-K LLM layers unfrozen
deepspeed --num_gpus=$NUM_DEVICES src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id "$MODEL_NAME" \
    --data_path /kaggle/input/600sample-real/llava_training_data.json \
    --image_folder /kaggle/input/train-zaic \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --unfreeze_topk_llm 6 \
    --lora_enable True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --bf16 False \
    --fp16 True \
    --disable_flash_attn2 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_min_pixels $((6 * 14 * 4 * 28 * 28)) \
    --video_max_pixels $((8 * 14 * 4 * 28 * 28)) \
    --fps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --logging_dir /kaggle/working/logs_phase2a \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --save_latest_only True \
    --dataloader_num_workers 0

echo "==========================================="
echo "✅ Phase 2a training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "==========================================="
