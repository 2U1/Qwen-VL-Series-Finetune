# Resume Training Guide

Hướng dẫn resume training và fine-tune thêm với config khác.

## Scenario 1: Resume với cùng config (sau khi bị timeout/crash)

Nếu training bị dừng giữa chừng và muốn continue với **cùng config**, chỉ cần chạy lại script:

```bash
bash scripts/train_zac_kaggle.sh
```

Script sẽ **TỰ ĐỘNG** resume từ checkpoint-latest nếu tìm thấy trong output_dir.

## Scenario 2: Resume và thay đổi config (Unfreeze LLM)

Nếu muốn **resume từ checkpoint cũ** nhưng **thay đổi config** (ví dụ: unfreeze LLM), làm theo các bước sau:

### Phase 1: Train với Freeze LLM (ĐÃ XONG)

```bash
# Config phase 1:
--freeze_vision_tower True
--freeze_llm True          # ← Frozen
--freeze_merger False
--num_train_epochs 3
--learning_rate 1e-4
```

✅ Checkpoint sau phase 1: `/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest`

### Phase 2: Resume và Unfreeze LLM (MỚI)

#### Option A: Dùng script có sẵn (KHUYẾN NGHỊ)

```bash
cd /kaggle/working/Qwen-VL-Series-Finetune
bash scripts/train_zac_kaggle_phase2.sh
```

Script này sẽ:
1. ✅ Copy checkpoint từ phase 1
2. ✅ Resume từ checkpoint đó
3. ✅ Unfreeze LLM (`freeze_llm=False`)
4. ✅ Giảm learning rate xuống 5e-5 (vì LLM đã pretrained tốt)
5. ✅ Giảm resolution để tránh OOM (vì LLM trainable tốn nhiều memory)
6. ✅ Train thêm 2 epochs
7. ✅ Save vào folder mới: `/kaggle/working/checkpoints/zac_qwen2vl_phase2`

#### Option B: Manual config

Nếu muốn tự customize, edit script:

```bash
# 1. Copy checkpoint to new directory
mkdir -p /kaggle/working/checkpoints/zac_qwen2vl_phase2
cp -r /kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest/* \
      /kaggle/working/checkpoints/zac_qwen2vl_phase2/

# 2. Run training với config mới
deepspeed src/train/train_sft.py \
    --model_id "Qwen/Qwen2-VL-2B-Instruct" \
    --output_dir /kaggle/working/checkpoints/zac_qwen2vl_phase2 \
    --freeze_llm False \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    # ... other args ...
```

## Những thay đổi quan trọng khi Unfreeze LLM

### 1. Learning Rate

```bash
# Phase 1 (freeze LLM): lr = 1e-4
# Phase 2 (unfreeze LLM): lr = 5e-5  # GIẢM 50%
```

**Lý do:** LLM đã được pretrain rất tốt, train với lr cao sẽ làm망 catastrophic forgetting.

### 2. Resolution

```bash
# Phase 1 (freeze LLM):
--video_min_pixels 263424   # 8 × 14 × 4 × 28 × 28
--video_max_pixels 395136   # 12 × 14 × 4 × 28 × 28

# Phase 2 (unfreeze LLM):
--video_min_pixels 197568   # 6 × 14 × 4 × 28 × 28  ← GIẢM
--video_max_pixels 296352   # 9 × 14 × 4 × 28 × 28  ← GIẢM
```

**Lý do:** LLM trainable → nhiều gradients hơn → tốn nhiều memory hơn → cần giảm resolution.

### 3. Number of Epochs

```bash
# Phase 1: 3 epochs (train LoRA + merger)
# Phase 2: 2 epochs (fine-tune LLM)
```

**Lý do:** LLM đã pretrained tốt, chỉ cần 1-2 epochs để adapt vào task.

### 4. Batch Size

```bash
# Nếu vẫn bị OOM, giảm batch size:
BATCH_PER_DEVICE=1  # Thay vì 2
```

### 5. Output Directory

```bash
# Phase 1: /kaggle/working/checkpoints/zac_qwen2vl_lora
# Phase 2: /kaggle/working/checkpoints/zac_qwen2vl_phase2  ← MỚI
```

**Lý do:** Tránh ghi đè checkpoint phase 1 (để có thể rollback nếu cần).

## Kiểm tra resume có hoạt động không

Khi chạy training, bạn sẽ thấy log:

```
Resuming from checkpoint: /kaggle/working/checkpoints/zac_qwen2vl_phase2
Loading merger weights from .../merger_weights.bin
Successfully loaded 6 merger parameters
```

Và training sẽ bắt đầu từ epoch 0 (của phase 2), không phải epoch 4.

**Lưu ý:** Khi resume, epoch counter sẽ reset về 0 cho phase mới. Để track tổng số epochs:
- Phase 1: 3 epochs
- Phase 2: 2 epochs
- **Tổng:** 5 epochs

## Expected Memory Usage

### Phase 1 (Freeze LLM):

```
Trainable params: 221M (9.23%)
- LoRA weights: ~215M
- Merger weights: ~6M
- LLM: 0 (frozen)
```

Memory: ~12GB VRAM per GPU

### Phase 2 (Unfreeze LLM):

```
Trainable params: ~1.5B (60-70%)
- LoRA weights: ~215M
- Merger weights: ~6M
- LLM LoRA: ~1.3B
```

Memory: ~16-18GB VRAM per GPU

→ Nếu GPU chỉ có 16GB, **PHẢI** giảm resolution hoặc batch size!

## Troubleshooting

### Lỗi: OOM khi unfreeze LLM

**Giải pháp:**

1. Giảm resolution thêm:
```bash
--video_min_pixels $((4 * 14 * 4 * 28 * 28))  # 131,712
--video_max_pixels $((6 * 14 * 4 * 28 * 28))  # 197,568
```

2. Giảm batch size:
```bash
BATCH_PER_DEVICE=1
```

3. Dùng gradient checkpointing (đã enable):
```bash
--gradient_checkpointing True  # ✅ Đã có
```

4. Offload optimizer sang CPU (sửa `scripts/zero2.json`):
```json
{
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  }
}
```

### Lỗi: Checkpoint không load được

**Nguyên nhân:** Output directory không có checkpoint

**Giải pháp:** Copy checkpoint trước khi train:
```bash
mkdir -p /kaggle/working/checkpoints/zac_qwen2vl_phase2
cp -r /kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest/* \
      /kaggle/working/checkpoints/zac_qwen2vl_phase2/
```

### Lỗi: Learning rate quá cao, loss tăng

**Triệu chứng:**
```
{'loss': 2.5, 'epoch': 0.1}  # Phase 1 end
{'loss': 5.8, 'epoch': 0.1}  # Phase 2 start - LOSS TĂNG!
```

**Giải pháp:** Giảm learning rate xuống:
```bash
--learning_rate 2e-5  # Hoặc thậm chí 1e-5
```

### Lỗi: Model quên kiến thức cũ (Catastrophic Forgetting)

**Triệu chứng:** Accuracy trên validation set **GIẢM** so với phase 1

**Giải pháp:**
1. Giảm learning rate xuống 1e-5 hoặc 2e-5
2. Giảm số epochs xuống 1 epoch
3. Tăng warmup_ratio lên 0.1
4. Consider dùng LoRA cho LLM thay vì unfreeze toàn bộ

## Best Practices

### 1. Incremental Unfreezing

Thay vì unfreeze toàn bộ LLM, có thể unfreeze từng phần:

```bash
# Phase 2a: Unfreeze top 6 layers
--unfreeze_topk_llm 6

# Phase 2b: Unfreeze all
--freeze_llm False
```

### 2. Learning Rate Scheduling

Dùng learning rate thấp hơn khi unfreeze:

```bash
# Phase 1: lr = 1e-4
# Phase 2: lr = 5e-5  (50%)
# Phase 3: lr = 2e-5  (20%)
```

### 3. Monitor Validation Accuracy

Track accuracy sau mỗi epoch để tránh overfitting:

```bash
# Add validation
--eval_path /path/to/val_data.json \
--evaluation_strategy epoch \
--eval_steps 74  # 1 epoch = 74 steps
```

### 4. Save Checkpoints của cả 2 phases

Giữ cả checkpoint phase 1 và phase 2:

```
checkpoints/
├── zac_qwen2vl_lora/          # Phase 1 (freeze LLM)
│   └── checkpoint-latest/
└── zac_qwen2vl_phase2/        # Phase 2 (unfreeze LLM)
    └── checkpoint-latest/
```

→ Nếu phase 2 không tốt, có thể rollback về phase 1.

## Example: Full 2-Phase Training Pipeline

```bash
# Phase 1: Train LoRA + Merger (3 epochs)
bash scripts/train_zac_kaggle.sh
# → Accuracy: 75%

# Evaluate phase 1
bash scripts/kaggle_evaluate.sh
# → Save results: phase1_results.json

# Phase 2: Unfreeze LLM (2 epochs)
bash scripts/train_zac_kaggle_phase2.sh
# → Accuracy: 82%  (expected improvement)

# Evaluate phase 2
python scripts/inference_mcq.py \
    --lora_path /kaggle/working/checkpoints/zac_qwen2vl_phase2/checkpoint-latest \
    --data_path /kaggle/input/600sample-real/llava_training_data.json \
    --video_folder /kaggle/input/train-zaic \
    --output phase2_results.json

# Compare results
python scripts/compare_phases.py phase1_results.json phase2_results.json
```

## Summary

| Aspect | Phase 1 (Freeze LLM) | Phase 2 (Unfreeze LLM) |
|--------|---------------------|------------------------|
| freeze_llm | True ✅ | False ✅ |
| Learning rate | 1e-4 | 5e-5 (↓50%) |
| Resolution | 263K-395K | 197K-296K (↓25%) |
| Epochs | 3 | 2 |
| Trainable params | 221M (9%) | ~1.5B (60%) |
| Memory/GPU | ~12GB | ~16-18GB |
| Expected accuracy | 70-80% | 75-85% |
| Training time | ~8-9h | ~10-12h |

**Khuyến nghị:**
- Luôn train phase 1 trước (freeze LLM) để model học được task basics
- Chỉ unfreeze LLM (phase 2) khi phase 1 đã cho accuracy > 70%
- Monitor validation accuracy để tránh overfitting
- Keep checkpoint của cả 2 phases để có thể rollback
