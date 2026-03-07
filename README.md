# Fine-tuning Qwen-VL Series

This repository contains a script for training [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
and [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) with only using HuggingFace and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).

## Other projects

**[[Phi3-Vision Finetuning]](https://github.com/2U1/Phi3-Vision-Finetune)**<br>
**[[Llama3.2-Vision Finetuning]](https://github.com/2U1/Llama3.2-Vision-Ft)**<br>
**[[Molmo Finetune]](https://github.com/2U1/Molmo-Finetune)**<br>
**[[Pixtral Finetune]](https://github.com/2U1/Pixtral-Finetune)**<br>
**[[SmolVLM Finetune]](https://github.com/2U1/SmolVLM-Finetune)**<br>
**[[Gemma3 Finetune]](https://github.com/2U1/Gemma3-Finetune)**

## Update

- [2026/03/07] Update codebase to `transformers==5.3.0`
- [2025/11/28] 🔥**Supports video training with DPO and GRPO.**
- [2025/11/27] 🔥**Supports Qwen3-VL-MoE**
- [2025/11/26] Update support for liger-kernel in Qwen3-VL.
- [2025/10/16] 🔥**Supports Qwen3-VL(non-moe)**
- [2025/08/21] Add option for using 2-layer mlp for classification.
- [2025/08/21] Add option for unfreeze only few layers for llm and vision tower.
- [2025/08/08] 🔥Monkey patch Qwen2.5-VL's window attention and forward for using less memory and speedups.
- [2025/07/25] Updated Classification training script for experimental feature.
- [2025/05/29] 🔥Supports GRPO training.
- [2025/04/16] 🔥Supports DPO training.
- [2025/03/04] Add Option for using liger kernel.
- [2025/02/18] 🔥Supports mixed-modality dataset with zero3.
- [2025/02/05] Fixed code for properly use image.
- [2025/02/03] Support Liger-kernel for Qwen2.5-VL.
- [2025/02/03] 🔥Supports Qwen2.5-VL.
- [2025/01/24] Add option for using DoRA.
- [2025/01/24] Fix error in LoRA training.
- [2025/01/18] 🔥Supports mixed-modality data.
- [2024/09/12] 🔥Now the model is trained using [Liger-Kernel](https://github.com/linkedin/Liger-Kernel).
- [2024/09/11] Supports setting different learning rates to projector and vision model.
- [2024/09/11] 🔥Supports multi-image and video training.

## Table of Contents

- [Fine-tuning Qwen-VL Series](#fine-tuning-qwen-vl-series)
  - [Other projects](#other-projects)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Docker](#docker)
  - [Installation](#installation)
    - [Environments](#environments)
    - [Using `requirements.txt`](#using-requirementstxt)
    - [Using `environment.yaml`](#using-environmentyaml)
  - [Dataset Preparation](#dataset-preparation)
  - [Supervised Fine Tuning](#supervised-fine-tuning)
    - [Full Finetuning](#full-finetuning)
    - [Finetune with LoRA](#finetune-with-lora)
    - [Train with video dataset](#train-with-video-dataset)
      - [Image Resolution for vram usage](#image-resolution-for-vram-usage)
      - [Merge LoRA Weights](#merge-lora-weights)
    - [Evaluation during Training](#evaluation-during-training)
      - [Step 1: Prepare Evaluation Dataset](#step-1-prepare-evaluation-dataset)
      - [Step 2: Define compute_metrics Function](#step-2-define-compute_metrics-function)
      - [Step 3: Modify Training Script](#step-3-modify-training-script)
      - [Step 4: Add Evaluation Arguments](#step-4-add-evaluation-arguments)
  - [DPO Finetuning](#dpo-finetuning)
  - [GRPO Finetuning](#grpo-finetuning)
    - [Prerequisites](#prerequisites)
  - [Classification Finetuning](#classification-finetuning)
    - [⚠️This is an experimental feature.](#️this-is-an-experimental-feature)
      - [Experimental Features](#experimental-features)
  - [Inference](#inference)
    - [Gradio Infernce (WebUI)](#gradio-infernce-webui)
  - [Issue for libcudnn error](#issue-for-libcudnn-error)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- LoRA/QLoRA
- Full-finetuning
- Enable finetuning `vision_model` while using LoRA
- Unfreeze only top-k layer
- Disable/enable Flash Attention 2
- Multi-image and video training
- Training optimized with liger kernel
- Mixed-modality dataset
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)

## Docker

To simplfy the setting process for training, you could use the provided pre-build environments.<br>
The settings are done in the conda env named `train`.<br><br>
You could find more information about the image [here](https://hub.docker.com/repository/docker/john119/vlm/general).

```
docker pull john119/vlm
docker run --gpus all -it -v /host/path:/docker/path --name vlm --ipc=host john119/vlm /bin/bash
```

## Installation

### Environments

- Ubuntu 22.04
- Nvidia-Driver 550.120
- Cuda version 12.8

Install the required packages using `environment.yaml`.

### Using `requirements.txt`

```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

### Using `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**When using a multi-image dataset, the image tokens should all be `<image>`, and the image file names should have been in a list.**<br><br>
**Please see the example below and follow format your data.**

<details>
<summary>Example for single image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
      {
        "from": "human",
        "value": "What feature can be seen on the back of the bus?"
      },
      {
        "from": "gpt",
        "value": "The back of the bus features an advertisement."
      },
      {
        "from": "human",
        "value": "Is the bus driving down the street or pulled off to the side?"
      },
      {
        "from": "gpt",
        "value": "The bus is driving down the street, which is crowded with people and other vehicles."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for multi image dataset</summary>

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera differnt?"
      },
      {
        "from": "gpt",
        "value": "Yes, It the perspective of the camera is different."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for video dataset</summary>

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
  ...
]
```

</details>

<details>
<summary>Example for DPO dataet</summary>

```json
[
  {
    "id": "e4cd017f-3854-4019-9cc1-9dacacb5d424",
    "image": "example_0_0.png",
    "prompt": "<image>\nDescribe the key features of the image in great detail.",
    "chosen": "The image features a modern, multi-colored train parked on a train track, occupying a significant portion of the scene. The train has a unique and striking design, with red, white, and blue colors, as well as yellow accents. It appears to be a high-speed train, indicating its advanced engineering and potential for fast travel.\n\nThe train is situated next to a platform, suggesting that it may be at a train station. The overall scene conveys a sense of transportation and modern technology, with the vibrant and sleek train as the main subject.",
    "rejected": "The image features a modern, colorful train parked on a set of train tracks. The train has a striking design, with red, white, and blue colors as well as yellow accents. It appears to be a high-speed train, ready for departure.\n\nThe train spans a significant portion of the image, from the left to the right side. The train tracks can be seen clearly beneath the train, emphasizing its position on the railway line. The overall scene gives off an impression of a contemporary and efficient mode of transportation."
  },
  {
    "id": "5e19e647-e5d3-4bcf-82e9-d262570743ae",
    "image": "example_1_0.png",
    "prompt": "<image>\nIs this bus in the USA?",
    "chosen": "Yes, based on the image, it can be assumed that this bus is in the USA. The location of the bus cannot be accurately determined.",
    "rejected": "No, it's not in the USA. The image does not provide specific information on where the bus is located. However, we can say that it's not in the United States."
  }
  ...
]
```

</details>

<details>
<summary>Example for GRPO dataset</summary>

```json
[
  {
    "id": "06bc8a17-bb1c-4007-8c08-92c41e2628b2",
    "image": "image_2.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nBased on the image, which geometric method is used to determine the bearing angle, and why is it the most appropriate choice?"
      },
      {
        "from": "gpt",
        "value": "<think>Let's analyze the image step-by-step. The image shows a right-angled triangle with points B, C, and A. The angle at point B is a right angle, indicating that trigonometric functions can be applied. To find the bearing angle, we need to relate the sides of the triangle. The tangent function is suitable here because it relates the opposite side (BC) to the adjacent side (AB) in a right-angled triangle. By using the tangent function, we can calculate the angle at point A, which is the bearing angle. Therefore, the most appropriate geometric method is the use of trigonometric functions.</think>\n\n<answer>A</answer>"
      }
    ]
  }
  ...
]
```

</details>

<br><br>

Adding the new domain-specific data on top of the general data from open-source data will enhance downstream capabilities while retaining the foundational skills. Of course, you can also choose to fine-tune solely on the new data based on your requirements.

## Supervised Fine Tuning

⚠️**For Qwen3-VL models, using liger-kernel with full fine-tuning is awfully slow. I recommend turning off liger-kernel or use zero2 with full-finetuning.**<br><br>

**Note:** Deepspeed zero2 is faster than zero3, however it consumes more memory. Also, most of the time zero2 is more stable than zero3.<br><br>
**Tip:** You could use `adamw_bnb_8bit` for optimizer to save memory.

To run the training script, use the following command:

### Full Finetuning

**Note:** If you want to use `unfreeze_topk_llm` or `unfreeze_topk_vision` you should set `-freeze_llm` or `--freeze_vision_tower` to `true`.

```bash
bash scripts/finetune.sh
```

### Finetune with LoRA

**Note:** Liger-kernel won't work with QLoRA. You need to disable to use QLoRA.<br>
**Note:** Do not combine quantization (`--bits 4` / `--bits 8`, i.e. QLoRA) with vision training. If you want to train vision-related modules (`--vision_lora True`, `--freeze_vision_tower False`, or `--unfreeze_topk_vision > 0`), use full precision training (`--bits 16`).<br>
If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

**IMPORTANT:** If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together.

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--use_liger` (bool): Option for using liger kernel to save memory.
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--freeze_merger` (bool): Option to tune projector (default: False).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input tokens for image.
- `--image_max_pixles` (int): Option for maximum maxmimum tokens for image.
- `--video_min_pixels` (int): Option for minimum input tokens for video.
- `--video_max_pixles` (int): Option for maximum maxmimum tokens for video.
- `--image_resized_width` (int): Option for setting the width of the input image.
- `--image_resized_height` (int): Option for setting the height of the input image.
- `--video_resized_width` (int): Option for setting the width of the input video.
- `--video_resized_height` (int): Option for setting the height of the input video.
- `--fps` (float): Frames per second for video data.
- `--nframes` (int): Number of frames for video data.
- `--unfreeze_topk_llm` (int): Number of top layers to unfreeze in the language model.
- `--unfreeze_topk_vision` (int): Number of top layers to unfreeze in the vision model.
- `--lora_enable` (bool): Option for using LoRA.
- `--vision_lora` (bool): Option for including `vision_tower` in LoRA module. `lora_enable` should be `True` to use this option.
- `--use_dora` (bool): Option for using DoRA instead of LoRA. `lora_enable` should be `True` to use this option.
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 128).
- `--lora_alpha` (int): LoRA alpha (default: 256).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

</details>

### Train with video dataset

You can train the model using a video dataset. You can set LoRA configs and use for LoRA too.<br>
**Note:** You could not set `fps` and `nframes` at the same time.

```bash
bash scripts/finetune_video.sh
```

**Note:** When training with video, it just as multi-image so you should adjust the `max_pixels` for maximum resolution and `fps` based on the available VRAM.

If you run out of vram, you can use [zero3_offload](./scripts/zero3_offload.json) instead of [zero3](./scripts/zero3_offload.json).<br>
You could use [zero2_offload](./scripts/zero2_offload.json) for a bit faster training.

#### Image Resolution for vram usage

The model supprots a wide range of resolution inputs. By default, it uses the native resolution for input.
For better performance using native or higer pixel numbers are recommended, however it takes too much memory and computation time for large images. So you could adjust the pixel numbers for it.
The model splits the image into `token * 28 * 28` so you could just change the the token_num part in the script. <br><br>
⚠️**For Qwen3-VL models, it should be `token * 32 * 32`.**<br><br>
For example:

```
--image_min_pixels $((256 * 28 * 28))
--image_max_pixels $((1280 * 28 * 28))
--video_min_pixels $((128 * 28 * 28))
--video_max_pixels $((768 * 28 * 28))
```

Besides you could directly set the image/video height and width to control over the memory.

```
--image_resized_width 448
--image_resized_height 448
--video_resized_width 448
--video_resized_height 448
```

These values will be rounded to the nearest multiple of 28.

#### Merge LoRA Weights

```
bash scripts/merge_lora.sh
```

**Note:** Remember to replace the paths in `finetune.sh` or `finetune_lora.sh` with your specific paths. (Also in `merge_lora.sh` when using LoRA.)

### Evaluation during Training

You can run generation-based evaluation during training by providing an evaluation dataset and a custom `compute_metrics` function. This allows you to monitor metrics like accuracy, BLEU, or any custom metric based on the model's generated text outputs.

#### Step 1: Prepare Evaluation Dataset

The evaluation dataset uses the same format as the training dataset. Place your evaluation data JSON file and specify the path using `--eval_path`.

```json
[
  {
    "id": "eval_001",
    "image": "test_image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat is shown in this image?"
      },
      {
        "from": "gpt",
        "value": "A cat sitting on a couch."
      }
    ]
  }
]
```

#### Step 2: Define compute_metrics Function

Create a custom `compute_metrics` function in your training script. The function receives a `GenerativeEvalPrediction` object containing:

- `predictions`: List of generated text strings from the model
- `references`: List of ground truth answer strings

```python
from src.trainer import GenerativeEvalPrediction

def compute_metrics(eval_pred: GenerativeEvalPrediction):
    predictions = eval_pred.predictions
    references = eval_pred.references

    # Example: Exact match accuracy
    correct = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    accuracy = correct / len(predictions) if predictions else 0

    return {"accuracy": accuracy}
```

#### Step 3: Modify Training Script

Update your training script (`src/train/train_sft.py`) to pass `compute_metrics` to the trainer:

```python
from src.trainer import QwenSFTTrainer, GenerativeEvalPrediction

def compute_metrics(eval_pred: GenerativeEvalPrediction):
    predictions = eval_pred.predictions
    references = eval_pred.references
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return {"accuracy": correct / len(predictions)}

# ... (model and data setup code)

trainer = QwenSFTTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    compute_metrics=compute_metrics,  # Add this line
    **data_module
)
```

#### Step 4: Add Evaluation Arguments

Add these arguments to your training script:

```bash
  --eval_path /path/to/eval.json \
  --eval_strategy steps \
  --eval_steps 500 \
  --per_device_eval_batch_size 1 \
  --generation_max_new_tokens 256 \
  --prediction_loss_only False \
  # ... other arguments
```

<details>
<summary>Evaluation Arguments</summary>

- `--eval_path` (str): Path to the evaluation data JSON file.
- `--eval_strategy` (str): Evaluation strategy - "steps" or "epoch" (default: "no").
- `--eval_steps` (int): Number of steps between evaluations (when eval_strategy="steps").
- `--per_device_eval_batch_size` (int): Batch size for evaluation (default: 8).
- `--generation_max_new_tokens` (int): Maximum new tokens to generate during evaluation (default: 512).
- `--prediction_loss_only` (bool): Set to False to enable generation-based evaluation (default: True).

</details>

<details>
<summary>Example: Custom Metrics with Multiple Scores</summary>

```python
from src.trainer import GenerativeEvalPrediction
import re

def compute_metrics(eval_pred: GenerativeEvalPrediction):
    predictions = eval_pred.predictions
    references = eval_pred.references

    # Exact match
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )

    # Contains match (reference appears in prediction)
    contains_matches = sum(
        1 for p, r in zip(predictions, references)
        if r.strip().lower() in p.strip().lower()
    )

    n = len(predictions)
    return {
        "exact_match": exact_matches / n if n > 0 else 0,
        "contains_match": contains_matches / n if n > 0 else 0,
    }
```

</details>

**Note:** Generation-based evaluation is slower than loss-only evaluation because it runs `model.generate()` for each sample. Consider using a smaller evaluation dataset or less frequent evaluation steps.

## DPO Finetuning

You can train the model using Direct Preference Optimization (DPO).<br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during DPO training just like in SFT.

```bash
bash scripts/finetune_dpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for DPO training.

<details>
<summary>Training arguments</summary>

- `--dpo_loss` (str): Loss type for dpo. (default: 'sigmoid')
- `--precompute_ref_log_probs` (bool): Wheter to precompute the reference log probs (default: False)
- `--beta` (float): The beta value for DPO (default: 0.1)

</details>

## GRPO Finetuning

You can traing the model using Group Relative Policy Optimization (GRPO) <br>
The process is quite similar to Supervised Fine-Tuning (SFT), and you can also apply LoRA during GRPO training just like in SFT.<br>
<br>

### Prerequisites

| What                      | Where                       | Notes                                                                                       |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **Reward functions**      | `src/train/reward_funcs.py` | Add any function that ends with `_reward`. The training script picks them up automatically. |
| **Custom system prompts** | `src/constants.py`          | Append your own prompt strings here.                                                        |

You could start training using this script.<br>
Before training, **Please check the dataset format once more.** The format is a bit different from other training methods.

```bash
bash scripts/finetune_grpo.sh
```

Most of the training arugments are same as SFT, but few other arguments are added for GRPO training.

<details>
<summary>Training arguments</summary>

- `--temperature` (float): Generation config (default: 0.9)
- `--top_p` (float): Generation config (default: 1.0)
- `--top_k` (int): Generation config (default: 50)
- `--min_p` (float): Generation config (default: None)
- `--repetition_penalty` (float): Generation config (default: 1.0)
- `--max_completion_length` (int): Max length for the completion (default: 256)
- `--max_prompt_length` (int): Max length for the prompt (default: 512)
- `--beta` (float): KL Coefficient. (default: 0.04)

</details>

## Classification Finetuning

### ⚠️This is an experimental feature.

The [model](src/model/modeling_cls.py) is tailored for classification tasks, such as other SequenceClassification models.

For the classification task, you need to prepare the dataset in a specific format. The dataset should be a JSON file where each entry contains an image and its corresponding label. The labels should be integers starting from 0.<br>
You can set the text in the filed `prompt` to provide a questions and options for the classification task. Also if your dataset dose not contain the `prompt` field, the script will automatically use the `USER_MESSAGE` from the [cls_dataset.py](src/dataset/cls_dataset.py).<br>

**Please see the example below for the dataset format.**<br>

<details>
<summary>Example for Classification Dataset</summary>

```json
[
  {
    "id": "06bc8a17-bb1c-4007-8c08-92c41e2628b2",
    "image": "image_2.jpg",
    "prompt": "Question: What is in the image? \n Options: \n 1. A train \n 2. A bus \n 3. A car \n 4. A bicycle",
    "label": "3",
  }
  ...
]
```

**Note:** You should set the `CLASS_2_ID` variable in the [cls_dataset.py](src/dataset/cls_dataset.py).

</details>

<br>

The dataset can contain **single/multi-image or video data**, and the model will be trained to classify the images/videos based on the provided labels.<br>

For now, you can select loss from one of the following:

- `cross_entropy`
- `focal_loss`
- `class_balanced_cross_entropy`
- `class_balanced_focal_loss`

Also you can set early stopping patience and threshold for the training.
For example, you can set `--early_stopping_patience 5` and `--early_stopping_threshold 0.01` to stop the training if the validation loss does not improve for 5 epochs with a threshold of 0.01.

Most of the training arugments are same as SFT, but few other arguments are added for classification training.<br><br>

**Tip:** In models like the Qwen family, which have strong context embeddings, even a shallow nonlinearity (a 1-layer MLP) can often improve separability in the tail. This works by introducing a bit of curvature that a purely linear head cannot provide.

<details>
<summary>Training arguments</summary>

- `--loss_type` (str): Loss type for classification (default: 'cross_entropy').
- `--focal_alpha` (str): Focal Loss alpha value. If None use CrossEntropyLoss. ex '1.0,7.5' (default: None).
- `--focal_gamma` (float): Focal Loss gamma value. (default: 0.0)
- `--num_labels` (int): Number of labels for classification
- `--class_balanced_beta` (float): Class Balanced beta value. (default: 0.999)
- `--early_stopping_patience` (int): Early stopping patience (default: 0)
- `--early_stopping_threshold` (float): Early stopping threshold (default: 0.01)
- `--mlp_head_dim` (int): Dimension of the MLP head (default: 0)
- `--mlp_head_dropout` (float): Dropout rate for the MLP head (default: 0.0)

</details>

You can run the training script using the following command:

```bash
bash scripts/finetune_cls.sh
```

#### Experimental Features

- Sampler for the dataset. The trainer scripts supports the sampler for the dataset. You could make your own sampler with inherting `DistributedSampler`.

## Inference

**Note:** You should use the merged weight when trained with LoRA.

### Gradio Infernce (WebUI)

1. Install gradio

```
pip install gradio
```

2. Launch app

```
python -m src.serve.app \
    --model-path /path/to/merged/weight
```

You can launch gradio based demo with this command. This can also set some other generation configs like `repetition_penalty`, `temperature` etc.

## Issue for libcudnn error

```
Could not load library libcudnn_cnn_train.so.8. Error: /usr/local/cuda-12.1/lib/libcudnn_cnn_train.so.8: undefined symbol: _ZN5cudnn3cnn34layerNormFwd_execute_internal_implERKNS_7backend11VariantPackEP11CUstream_stRNS0_18LayerNormFwdParamsERKNS1_20NormForwardOperationEmb, version libcudnn_cnn_infer.so.8
```

You could run `unset LD_LIBRARY_PATH` for this error.
You could see this [issue](https://github.com/andimarafioti/florence2-finetuning/issues/2)

## TODO

- [x] Support for video data
- [x] Add demo for multi-image and video
- [x] Handle mixed-modality data in dataset and collator
- [x] Support Qwen2.5-VL
- [x] Monkey-patch liger-kernel for Qwen2.5-VL
- [x] Update the code base to the latest transformers.
- [x] Add DPO
- [x] Add GRPO
- [x] Support Qwen3-VL(non-moe)
- [x] Support Qwen3-VL-Moe

## Known Issues

- [libcudnn issue](#issue-for-libcudnn-error)

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{Qwen2-VL-Finetuning,
  author = {Yuwon Lee},
  title = {Qwen2-VL-Finetune},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/2U1/Qwen2-VL-Finetune}
}
```

## Acknowledgement

This project is based on

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT): An amazing open-source project of LMM.
- [Mipha](https://github.com/zhuyiche/llava-phi): Open-source projcet of SMM with amazing capabilites.
- [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Awesome pretrained MLLM based on Qwen2.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel): Collection of Tirton kernels designed specifically for LLM training.
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1): Open-source project of Reinforcement Learning with VLMs.
