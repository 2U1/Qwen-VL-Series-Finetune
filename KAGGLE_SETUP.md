# Setup hướng dẫn cho Kaggle

## Bước 1: Clone repo trong Kaggle Notebook

```bash
cd /kaggle/working
git clone https://github.com/thangquocdang/Qwen-VL-Series-Finetune.git
cd Qwen-VL-Series-Finetune
```

## Bước 2: Cài đặt dependencies

```bash
pip install -q -r requirements.txt
```

## Bước 3: Kiểm tra GPU

```bash
nvidia-smi
```

## Bước 4: Chuẩn bị dữ liệu

Đảm bảo bạn đã thêm datasets vào Kaggle notebook:
- Input: `zac-sample-600/zac_llava_format.json`
- Input: `train-zaic/videos`

Hoặc sửa đường dẫn trong `scripts/train_zac_kaggle.sh`:
```bash
--data_path /kaggle/input/YOUR_DATASET/zac_llava_format.json \
--image_folder /kaggle/input/YOUR_DATASET/videos \
```

## Bước 5: Chạy training

```bash
bash scripts/train_zac_kaggle.sh
```

## Bước 6: Monitor training

Trong terminal khác (hoặc cell khác):
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Monitor logs
tail -f /kaggle/working/logs/events.out.tfevents.*
```

## Bước 7: Lấy checkpoint sau khi training

Checkpoints sẽ được lưu tại:
```
/kaggle/working/checkpoints/zac_qwen2vl_lora/
├── checkpoint-50/
├── checkpoint-100/
├── checkpoint-150/
└── ...
```

## Các vấn đề thường gặp

### 1. ModuleNotFoundError: No module named 'src'

**Nguyên nhân:** PYTHONPATH không được set

**Giải pháp:** Script đã tự động set PYTHONPATH. Nếu vẫn lỗi, chạy thủ công:
```bash
export PYTHONPATH=/kaggle/working/Qwen-VL-Series-Finetune:$PYTHONPATH
```

### 2. Division by zero

**Nguyên nhân:** Biến môi trường không được set đúng

**Giải pháp:** Chạy script với bash (không phải sh):
```bash
bash scripts/train_zac_kaggle.sh  # ✅ Đúng
sh scripts/train_zac_kaggle.sh    # ❌ Sai
```

### 3. Out of Memory (OOM)

**Nguyên nhân:** GPU memory không đủ

**Giải pháp:** Giảm batch size hoặc resolution:
```bash
# Sửa trong scripts/train_zac_kaggle.sh
GLOBAL_BATCH_SIZE=2  # Giảm từ 4 xuống 2
```

### 4. Checkpoint không được lưu

**Nguyên nhân:** Process bị kill trước khi lưu

**Giải pháp:** Script đã được config để lưu mỗi 50 steps. Kiểm tra:
```bash
ls -lh /kaggle/working/checkpoints/zac_qwen2vl_lora/
```

## Configuration chính

| Parameter | Value | Mô tả |
|-----------|-------|-------|
| Model | Qwen2-VL-2B-Instruct | Base model |
| LoRA rank | 128 | Kích thước LoRA adapter |
| Batch size | 4 | Global batch size |
| Learning rate | 1e-4 | Learning rate |
| Epochs | 5 | Số epochs |
| Save steps | 50 | Lưu checkpoint mỗi 50 steps |

## Tùy chỉnh configuration

Sửa file `scripts/train_zac_kaggle.sh`:

```bash
# Thay đổi model
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Thay đổi batch size
GLOBAL_BATCH_SIZE=2

# Thay đổi learning rate
--learning_rate 2e-4 \

# Thay đổi số epochs
--num_train_epochs 10 \

# Thay đổi resolution
--video_min_pixels $((10 * 14 * 4 * 28 * 28)) \
--video_max_pixels $((12 * 14 * 4 * 28 * 28)) \
```
