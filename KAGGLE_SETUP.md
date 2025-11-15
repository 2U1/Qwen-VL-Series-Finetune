# Setup hướng dẫn cho Kaggle

## ⚠️ Lựa chọn số GPU

**KHUYẾN NGHỊ: Dùng 1 GPU** (script: `train_zac_kaggle.sh`)
- Nhanh hơn với dataset nhỏ
- Tránh OOM
- Không có communication overhead

**Nếu muốn thử 2 GPUs** (script: `train_zac_kaggle_2gpu.sh`)
- Resolution phải giảm thấp hơn
- Dễ bị OOM
- Có thể chậm hơn do synchronization

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

**QUAN TRỌNG:** Copy videos vào `/kaggle/working/` để tránh I/O bottleneck:

```bash
# Copy videos từ input vào working directory
cp -r /kaggle/input/train-zaic/videos /kaggle/working/
```

Đảm bảo bạn đã thêm datasets vào Kaggle notebook:
- Input: `zac-sample-600/zac_llava_format.json`
- Videos: Copy vào `/kaggle/working/videos`

## Bước 5: Chạy training

### Option A: 1 GPU (KHUYẾN NGHỊ)
```bash
bash scripts/train_zac_kaggle.sh
```

### Option B: 2 GPUs (experimental)
```bash
bash scripts/train_zac_kaggle_2gpu.sh
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

### 3. Out of Memory (OOM) - Process killed with return code -9

**Nguyên nhân:** GPU memory không đủ

**Triệu chứng:**
```
exits with return code = -9
WARNING: The given max_pixels[702464] exceeds limit[645388]
```

**Giải pháp:**

**A. Nếu dùng 1 GPU (khuyến nghị):**
```bash
# Trong scripts/train_zac_kaggle.sh
BATCH_PER_DEVICE=1  # Giảm từ 2 xuống 1
GLOBAL_BATCH_SIZE=4  # Giảm từ 8 xuống 4

# Hoặc giảm resolution
--video_min_pixels $((8 * 14 * 4 * 28 * 28))   # Giảm từ 10 xuống 8
--video_max_pixels $((12 * 14 * 4 * 28 * 28))  # Giảm từ 16 xuống 12
```

**B. Nếu dùng 2 GPUs:**
```bash
# 2 GPUs tốn gấp đôi memory!
# Phải dùng resolution thấp hơn nhiều
# Khuyến nghị: Chuyển sang 1 GPU thay vì giảm quality
```

**C. Kiểm tra resolution có hợp lệ:**
```bash
# MAX_PIXELS LIMIT = 645,388
# Tính toán: N * 14 * 4 * 28 * 28 < 645,388
# → N < 16.5

# Ví dụ hợp lệ:
--video_max_pixels $((16 * 14 * 4 * 28 * 28 - 10000))  # = 635,376 ✅
--video_max_pixels $((17 * 14 * 4 * 28 * 28))          # = 702,464 ❌
```

### 4. Disk Full - File write failed khi lưu checkpoint

**Nguyên nhân:** Kaggle working directory hết dung lượng (~20GB)

**Triệu chứng:**
```
RuntimeError: PytorchStreamWriter failed writing file data/XXX: file write failed
RuntimeError: unexpected pos XXXXXXX vs XXXXXXX
```

**Nguyên nhân chi tiết:**
- Mỗi checkpoint lưu `non_lora_state_dict.bin` (~500MB)
- Với 2 GPUs: 2 processes × 500MB = 1GB/checkpoint
- Videos trong `/kaggle/working/` chiếm nhiều dung lượng
- Multiple checkpoints × 1GB = Disk full!

**Giải pháp:**

**A. Mặc định - Không lưu non_lora_state_dict (Đã fix):**
```bash
# Scripts đã được update để skip file này
# Chỉ lưu LoRA weights (nhỏ hơn nhiều)
# Bạn không cần làm gì cả!
```

**B. Nếu VẪN hết disk:**
```bash
# 1. Tăng save_steps để lưu ít checkpoint hơn
--save_steps 50  # Thay vì 30

# 2. Giảm save_total_limit
--save_total_limit 1  # Chỉ giữ 1 checkpoint mới nhất

# 3. Kiểm tra disk space trước khi train
df -h /kaggle/working

# 4. Xóa checkpoint cũ nếu cần
rm -rf /kaggle/working/checkpoints/*/checkpoint-*
```

**C. Nếu MUỐN lưu non_lora_state_dict (không khuyến nghị):**
```bash
# Thêm flag này vào script
--save_non_lora_weights True

# Và đảm bảo có đủ dung lượng (ít nhất 5GB trống)
```

### 5. Checkpoint không được lưu

**Nguyên nhân:** Process bị kill trước khi lưu

**Giải pháp:** Script đã được config để lưu thường xuyên. Kiểm tra:
```bash
ls -lh /kaggle/working/checkpoints/zac_qwen2vl_lora/
```

## Resume Training

### Những gì được lưu trong mỗi checkpoint:

```
checkpoint-30/
├── adapter_model.bin       # LoRA weights (~150MB) ✅
├── adapter_config.json     # LoRA config ✅
├── merger_weights.bin      # Merger weights (~20MB) ✅ NEW!
├── optimizer.pt           # Optimizer state ✅
├── scheduler.pt           # Scheduler state ✅
├── trainer_state.json     # Training progress ✅
├── rng_state.pth         # Random states ✅
└── config.json           # Model config ✅
```

**Tổng dung lượng:** ~200MB/checkpoint (so với ~1GB trước đây!)

### Resume training tự động:

Training sẽ **TỰ ĐỘNG resume** nếu tìm thấy checkpoint:

```bash
# Chỉ cần chạy lại script
bash scripts/train_zac_kaggle.sh

# Output sẽ hiện:
# "Resuming from checkpoint: /kaggle/working/checkpoints/.../checkpoint-30"
# "Loading merger weights from .../merger_weights.bin"
# "Successfully loaded 6 merger parameters"
```

### Resume thủ công từ checkpoint cụ thể:

```python
# Trong Python notebook
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 1. Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16
)

# 2. Load LoRA weights
model = PeftModel.from_pretrained(
    model,
    "/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-30"
)

# 3. Load merger weights (nếu có)
merger_path = "/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-30/merger_weights.bin"
if os.path.exists(merger_path):
    merger_weights = torch.load(merger_path)
    model.load_state_dict(merger_weights, strict=False)

# 4. Load processor
processor = AutoProcessor.from_pretrained(
    "/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-30"
)
```

### Files CẦN giữ lại để resume:

**Tối thiểu (chỉ inference):**
- ✅ `adapter_model.bin` - LoRA weights
- ✅ `adapter_config.json` - LoRA config
- ✅ `merger_weights.bin` - Merger weights (nếu train merger)
- ✅ `config.json` - Model config

**Đầy đủ (resume training):**
- ✅ Tất cả files trên
- ✅ `optimizer.pt` - Để tiếp tục optimize đúng
- ✅ `scheduler.pt` - Để LR schedule đúng
- ✅ `trainer_state.json` - Để biết đang ở step nào

### Files KHÔNG cần thiết:

- ❌ `non_lora_state_dict.bin` (~500MB) - Đã được thay bằng `merger_weights.bin`
- ❌ Checkpoint cũ - Chỉ cần giữ 1-2 checkpoint mới nhất

## Configuration chính

### So sánh 1 GPU vs 2 GPUs

| Parameter | 1 GPU (Khuyến nghị) | 2 GPUs (Experimental) |
|-----------|---------------------|----------------------|
| Script | `train_zac_kaggle.sh` | `train_zac_kaggle_2gpu.sh` |
| Model | Qwen2-VL-2B-Instruct | Qwen2-VL-2B-Instruct |
| Số GPUs | 1 | 2 |
| Batch per GPU | 2 | 1 |
| Global Batch Size | 8 | 4 |
| Grad Accumulation | 4 | 2 |
| LoRA rank | 128 | 128 |
| Learning rate | 1e-4 | 1e-4 |
| Epochs | 3 | 3 |
| video_min_pixels | 360,448 | 263,424 |
| video_max_pixels | 635,376 | 329,280 |
| Save steps | 30 | 30 |
| DeepSpeed config | zero2.json | zero2_offload.json |
| Est. training time | ~8-9h | ~9-10h |
| OOM risk | Thấp ✅ | Cao ⚠️ |
| Video quality | Cao ✅ | Thấp ❌ |

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
