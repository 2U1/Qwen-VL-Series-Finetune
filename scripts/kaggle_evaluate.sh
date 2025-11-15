#!/bin/bash

# ===== Evaluate MCQ performance on Kaggle =====
# Calculates accuracy on validation/test set
# ==============================================

export PYTHONPATH=/kaggle/working/Qwen-VL-Series-Finetune:$PYTHONPATH
cd /kaggle/working/Qwen-VL-Series-Finetune

# Configuration
BASE_MODEL="Qwen/Qwen2-VL-2B-Instruct"
CHECKPOINT_DIR="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
DATA_PATH="/kaggle/input/600sample-real/llava_training_data.json"
VIDEO_FOLDER="/kaggle/input/train-zaic"
OUTPUT_PATH="/kaggle/working/evaluation_results.json"

echo "==========================================="
echo "Evaluating MCQ Performance"
echo "==========================================="
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "==========================================="

# Run evaluation
python scripts/inference_mcq.py \
    --base_model "$BASE_MODEL" \
    --lora_path "$CHECKPOINT_DIR" \
    --data_path "$DATA_PATH" \
    --video_folder "$VIDEO_FOLDER" \
    --output "$OUTPUT_PATH"

echo "==========================================="
echo "✅ Evaluation completed!"
echo "Results saved to: $OUTPUT_PATH"
echo "==========================================="
