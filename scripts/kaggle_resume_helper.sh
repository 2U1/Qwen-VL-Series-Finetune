#!/bin/bash

# ===== Kaggle Training Resume Helper =====
# Use this to manage multi-session training on Kaggle
# ==========================================

CHECKPOINT_DIR="/kaggle/working/checkpoints/zac_qwen2vl_lora"
OUTPUT_DIR="/kaggle/working/outputs"  # For uploading to Kaggle datasets

# Function to save checkpoint to Kaggle output
save_checkpoint_to_output() {
    echo "==========================================="
    echo "Saving checkpoint to Kaggle output..."
    echo "==========================================="

    # Find latest checkpoint (check for checkpoint-latest first, then numbered)
    if [ -d "${CHECKPOINT_DIR}/checkpoint-latest" ]; then
        LATEST="${CHECKPOINT_DIR}/checkpoint-latest"
    else
        LATEST=$(ls -td ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | head -1)
    fi

    if [ -z "$LATEST" ] || [ ! -d "$LATEST" ]; then
        echo "❌ No checkpoint found!"
        return 1
    fi

    echo "Latest checkpoint: $LATEST"

    # Create output directory
    mkdir -p ${OUTPUT_DIR}

    # Get current step from trainer_state.json
    if [ -f "${LATEST}/trainer_state.json" ]; then
        GLOBAL_STEP=$(python3 -c "import json; f=open('${LATEST}/trainer_state.json'); d=json.load(f); print(d.get('global_step', 'unknown'))" 2>/dev/null || echo "unknown")
        CHECKPOINT_NAME="checkpoint-${GLOBAL_STEP}"
    else
        CHECKPOINT_NAME=$(basename $LATEST)
    fi

    mkdir -p ${OUTPUT_DIR}/${CHECKPOINT_NAME}

    echo "Copying essential files..."
    cp ${LATEST}/adapter_model.bin ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ 2>/dev/null || echo "No adapter_model.bin"
    cp ${LATEST}/adapter_config.json ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ 2>/dev/null || echo "No adapter_config.json"
    cp ${LATEST}/merger_weights.bin ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ 2>/dev/null || echo "No merger_weights.bin"
    cp ${LATEST}/config.json ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ 2>/dev/null || echo "No config.json"
    cp ${LATEST}/trainer_state.json ${OUTPUT_DIR}/${CHECKPOINT_NAME}/ 2>/dev/null || echo "No trainer_state.json"

    # Show sizes
    echo "==========================================="
    echo "Checkpoint saved to: ${OUTPUT_DIR}/${CHECKPOINT_NAME}"
    du -sh ${OUTPUT_DIR}/${CHECKPOINT_NAME}
    echo "==========================================="
    echo "✅ Upload this to Kaggle Datasets for next session!"
    echo "==========================================="
}

# Function to check training progress
check_progress() {
    echo "==========================================="
    echo "Training Progress Check"
    echo "==========================================="

    if [ -f "${CHECKPOINT_DIR}/trainer_state.json" ]; then
        # Extract current step from trainer_state.json
        GLOBAL_STEP=$(python3 -c "import json; f=open('${CHECKPOINT_DIR}/trainer_state.json'); d=json.load(f); print(d.get('global_step', 0))")
        EPOCH=$(python3 -c "import json; f=open('${CHECKPOINT_DIR}/trainer_state.json'); d=json.load(f); print(round(d.get('epoch', 0), 2))")

        echo "Current Global Step: $GLOBAL_STEP"
        echo "Current Epoch: $EPOCH"
    fi

    # List checkpoints
    echo ""
    echo "Available checkpoints:"
    ls -lh ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | awk '{print $9, $5}'

    echo "==========================================="
}

# Function to clean old checkpoints
clean_old_checkpoints() {
    echo "==========================================="
    echo "Cleaning old checkpoints..."
    echo "==========================================="

    # Keep only the latest checkpoint
    ls -td ${CHECKPOINT_DIR}/checkpoint-* | tail -n +2 | xargs rm -rf

    echo "✅ Old checkpoints removed"
    df -h /kaggle/working
    echo "==========================================="
}

# Main menu
case "${1}" in
    save)
        save_checkpoint_to_output
        ;;
    check)
        check_progress
        ;;
    clean)
        clean_old_checkpoints
        ;;
    *)
        echo "Usage: $0 {save|check|clean}"
        echo ""
        echo "Commands:"
        echo "  save  - Save latest checkpoint to /kaggle/working/outputs"
        echo "  check - Check training progress"
        echo "  clean - Remove old checkpoints (keep latest)"
        echo ""
        exit 1
        ;;
esac
