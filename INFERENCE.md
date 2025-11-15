# Inference Guide - Qwen2-VL LoRA

Hướng dẫn load model đã train và chạy inference/validation.

## Cấu trúc Checkpoint

Sau khi train xong, checkpoint của bạn có cấu trúc:

```
checkpoints/zac_qwen2vl_lora/checkpoint-latest/
├── adapter_model.bin       # LoRA weights (~150MB)
├── adapter_config.json     # LoRA config
├── merger_weights.bin      # Merger weights (~20MB) - QUAN TRỌNG!
├── optimizer.pt           # Optimizer state (không cần cho inference)
├── scheduler.pt           # Scheduler state (không cần cho inference)
└── trainer_state.json     # Training progress
```

**Lưu ý quan trọng:**
- `adapter_model.bin`: LoRA weights cho language model
- `merger_weights.bin`: Weights của merger layer (vì bạn train với `freeze_merger=False`)
- Cả 2 files này đều cần thiết để load model đúng!

## Cách 1: Inference trên 1 video

### Trên Kaggle:

```python
import sys
sys.path.append('/kaggle/working/Qwen-VL-Series-Finetune')

from scripts.inference import load_model_with_lora, inference_video

# Load model
processor, model = load_model_with_lora(
    base_model_path="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
)

# Run inference
result = inference_video(
    processor,
    model,
    video_path="/kaggle/input/train-zaic/video_001.mp4",
    question="Hành vi của người trong video là gì?"
)

print(result)
```

### Từ command line:

```bash
cd /kaggle/working/Qwen-VL-Series-Finetune

python scripts/inference.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --lora_path /kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest \
    --video /kaggle/input/train-zaic/video_001.mp4 \
    --question "Hành vi của người trong video là gì?"
```

## Cách 2: Batch Inference trên toàn bộ dataset

### Sử dụng script có sẵn:

```bash
cd /kaggle/working/Qwen-VL-Series-Finetune
bash scripts/kaggle_inference.sh
```

Script này sẽ:
1. Load model từ checkpoint-latest
2. Chạy inference trên toàn bộ dataset
3. Save kết quả vào `/kaggle/working/predictions.json`

### Hoặc chạy thủ công:

```bash
python scripts/inference.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --lora_path /kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest \
    --data_path /kaggle/input/600sample-real/llava_training_data.json \
    --video_folder /kaggle/input/train-zaic \
    --output /kaggle/working/predictions.json
```

### Test trên một vài samples trước:

```bash
python scripts/inference.py \
    --base_model Qwen/Qwen2-VL-2B-Instruct \
    --lora_path /kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest \
    --data_path /kaggle/input/600sample-real/llava_training_data.json \
    --video_folder /kaggle/input/train-zaic \
    --output /kaggle/working/predictions_test.json \
    --max_samples 10  # Chỉ chạy 10 samples đầu
```

## Cách 3: Load model trong Python notebook

```python
import torch
import sys
sys.path.append('/kaggle/working/Qwen-VL-Series-Finetune')

from scripts.inference import load_model_with_lora, inference_video

# Load model với LoRA + merger weights
processor, model = load_model_with_lora(
    base_model_path="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
)

# Giờ bạn có thể dùng model như bình thường
# Model đã có LoRA weights + merger weights loaded

# Example: Inference trên 1 video
video_path = "/kaggle/input/train-zaic/video_001.mp4"
question = "Mô tả hành vi của người trong video"

result = inference_video(
    processor,
    model,
    video_path,
    question,
    max_new_tokens=256
)

print(f"Prediction: {result}")
```

## Cách 4: Merge LoRA weights vào base model (Optional)

Nếu bạn muốn tạo 1 model độc lập (không cần load LoRA mỗi lần):

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

# 1. Load base model
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
)

# 3. Load merger weights
merger_weights = torch.load(
    "/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest/merger_weights.bin",
    map_location="cpu"
)

for name, param in merger_weights.items():
    clean_name = name.replace('base_model.model.', '')
    if clean_name in model.state_dict():
        model.state_dict()[clean_name].copy_(param)

# 4. Merge LoRA into base model
merged_model = model.merge_and_unload()

# 5. Save merged model
merged_model.save_pretrained("/kaggle/working/merged_model")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor.save_pretrained("/kaggle/working/merged_model")

print("✅ Merged model saved to /kaggle/working/merged_model")
```

Sau đó bạn có thể load model merged:

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/kaggle/working/merged_model",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

## Format Output

File `predictions.json` có format:

```json
[
  {
    "id": "sample_001",
    "video": "video_001.mp4",
    "question": "Hành vi của người trong video là gì?",
    "prediction": "Người trong video đang...",
    "ground_truth": "Normal"
  },
  ...
]
```

## Troubleshooting

### Lỗi: "No such file 'merger_weights.bin'"

**Nguyên nhân:** Checkpoint được tạo với config cũ (trước khi thêm merger weights saving)

**Giải pháp:**
```python
# Load model mà không cần merger weights
processor, model = load_model_with_lora(
    base_model_path="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest"
)
# Script sẽ warning nhưng vẫn load được LoRA weights
```

### Lỗi: "Expected all tensors to be on the same device"

**Nguyên nhân:** Model được load trên multi-GPU và LoRA weights bị phân tán trên các GPU khác nhau

**Giải pháp:** Load model lên 1 GPU cụ thể (mặc định là cuda:0):

```python
# Python
processor, model = load_model_with_lora(
    base_model_path="Qwen/Qwen2-VL-2B-Instruct",
    lora_path="/kaggle/working/checkpoints/zac_qwen2vl_lora/checkpoint-latest",
    device="cuda:0"  # Force single GPU
)

# Command line (đã mặc định cuda:0)
python scripts/inference.py \
    --lora_path /path/to/checkpoint \
    --video /path/to/video.mp4 \
    --device cuda:0  # Optional, default is cuda:0
```

### Lỗi: Out of Memory khi inference

**Giải pháp:**
1. Giảm batch size trong batch inference
2. Load model với 8-bit quantization:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Performance không tốt

**Kiểm tra:**
1. Merger weights đã được load chưa? (Xem log khi load model)
2. Model có đang ở eval mode? (`model.eval()`)
3. Thử tăng `max_new_tokens` nếu output bị cắt ngắn

## Next Steps

Sau khi có predictions, bạn có thể:

1. **Tính metrics:** Accuracy, F1, etc.
2. **Visualize results:** So sánh prediction vs ground truth
3. **Error analysis:** Tìm cases model predict sai
4. **Upload to Kaggle:** Submit predictions nếu đây là competition

```python
import json
import pandas as pd

# Load predictions
with open('/kaggle/working/predictions.json') as f:
    preds = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(preds)
print(df.head())

# Calculate accuracy (if ground_truth available)
df['correct'] = df['prediction'] == df['ground_truth']
accuracy = df['correct'].mean()
print(f"Accuracy: {accuracy:.2%}")
```
