#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="ByteDance-Seed/UI-TARS-1.5-7B"

# ============================================================================
# SETUP: Choose configuration (A100 or L4)
# ============================================================================
SETUP="${SETUP:-A100}"  # Default to A100, override with: SETUP=L4 bash script.sh

if [ "$SETUP" == "A100" ]; then
    # A100 Single GPU Configuration
    NUM_DEVICES=1
    BATCH_PER_DEVICE=8
    GLOBAL_BATCH_SIZE=128
    GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
    DEEPSPEED_CONFIG="scripts/zero3_offload.json"
    
    # Update these paths for your GCP data disk
    DATA_PATH="${DATA_PATH:-/path/to/train_split.json}"
    IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/images}"
    EVAL_PATH="${EVAL_PATH:-/path/to/val_split.json}"  # Validation split (NOT test set)
    EVAL_IMAGE_FOLDER="${EVAL_IMAGE_FOLDER:-$IMAGE_FOLDER}"
    OUTPUT_DIR="${OUTPUT_DIR:-output/uitars_sft_a100}"
    
    # WandB configuration
    export WANDB_PROJECT="${WANDB_PROJECT:-uitars-sft}"
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-uitars-1.5-7b-lora-a100}"
    export WANDB_LOG_MODEL="false"
    
    NUM_EPOCHS=3
    LOGGING_STEPS=10
    REPORT_TO="wandb"
else
    # L4 Multi-GPU Configuration (original test setup)
    NUM_DEVICES=4
    BATCH_PER_DEVICE=1
    GLOBAL_BATCH_SIZE=128
    GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))
    DEEPSPEED_CONFIG="scripts/zero3_offload.json"
    
    DATA_PATH="src/data/training_data.json"
    IMAGE_FOLDER="src/data"
    OUTPUT_DIR="output/testing_lora"
    
    NUM_EPOCHS=3
    LOGGING_STEPS=1
    REPORT_TO="wandb"
fi

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Build eval arguments only if eval_path is set
EVAL_ARGS=""
if [ -n "$EVAL_PATH" ]; then
    EVAL_ARGS="--eval_path $EVAL_PATH"
    if [ -n "$EVAL_IMAGE_FOLDER" ]; then
        EVAL_ARGS="$EVAL_ARGS --eval_image_folder $EVAL_IMAGE_FOLDER"
    fi
    # Eval batch size: Use smaller than training to avoid OOM
    # During eval, model needs memory for forward pass + activations (no gradient checkpointing)
    # ZeRO-3 offload still brings parameters to GPU during forward pass
    # Default to half of training batch size, can override: EVAL_BATCH_SIZE=8 bash script.sh
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$((BATCH_PER_DEVICE / 2))}"
    # Ensure minimum of 1
    if [ "$EVAL_BATCH_SIZE" -lt 1 ]; then
        EVAL_BATCH_SIZE=1
    fi
    EVAL_ARGS="$EVAL_ARGS --eval_strategy epoch --per_device_eval_batch_size $EVAL_BATCH_SIZE"
fi

# Add max_train_samples if specified (for fair comparison across data variants)
MAX_SAMPLES_ARGS="6500"
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    MAX_SAMPLES_ARGS="--max_train_samples $MAX_TRAIN_SAMPLES"
fi

deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --num_lora_modules 32 \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_id $MODEL_NAME \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    $EVAL_ARGS \
    $MAX_SAMPLES_ARGS \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((16384 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps $LOGGING_STEPS \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to $REPORT_TO \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True