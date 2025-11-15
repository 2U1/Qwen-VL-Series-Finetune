#!/usr/bin/env python3
"""
Inference script for Qwen2-VL with LoRA adapters
Loads LoRA weights and merger weights for inference
"""

import torch
import argparse
from pathlib import Path
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json


def load_model_with_lora(
    base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: str = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    Load Qwen2-VL model with LoRA adapter and merger weights

    Args:
        base_model_path: Path to base model (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        lora_path: Path to checkpoint with LoRA weights (e.g., "checkpoints/.../checkpoint-latest")
        device: Device to load model on
        dtype: Data type for model weights
    """
    print(f"Loading base model from {base_model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    if lora_path:
        lora_path = Path(lora_path)

        # Load LoRA adapter
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(
            model,
            str(lora_path),
            is_trainable=False
        )

        # Load merger weights if available
        merger_weights_path = lora_path / "merger_weights.bin"
        if merger_weights_path.exists():
            print(f"Loading merger weights from {merger_weights_path}...")
            merger_weights = torch.load(merger_weights_path, map_location="cpu")

            # Load merger weights into model
            model_state = model.state_dict()
            for name, param in merger_weights.items():
                # Remove 'base_model.model.' prefix if present
                clean_name = name.replace('base_model.model.', '')
                if clean_name in model_state:
                    model_state[clean_name].copy_(param)
                    print(f"  Loaded: {clean_name}")

            print(f"Successfully loaded {len(merger_weights)} merger parameters")
        else:
            print("⚠️  No merger_weights.bin found, skipping merger weight loading")

    model.eval()
    print("✅ Model loaded successfully!")
    return processor, model


def inference_video(
    processor,
    model,
    video_path: str,
    question: str = "Describe what is happening in this video.",
    max_new_tokens: int = 256
):
    """
    Run inference on a video file

    Args:
        processor: Qwen2VL processor
        model: Loaded model
        video_path: Path to video file
        question: Question to ask about the video
        max_new_tokens: Maximum tokens to generate
    """
    # Prepare conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,  # Match training fps
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    # Generate
    print(f"\n🎬 Video: {video_path}")
    print(f"❓ Question: {question}")
    print("💭 Generating answer...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistent results
        )

    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"✅ Answer: {output_text}\n")
    return output_text


def batch_inference(
    processor,
    model,
    data_path: str,
    video_folder: str,
    output_path: str = None,
    max_samples: int = None
):
    """
    Run inference on a dataset (LLaVA format)

    Args:
        processor: Qwen2VL processor
        model: Loaded model
        data_path: Path to JSON data file
        video_folder: Folder containing videos
        output_path: Path to save predictions (optional)
        max_samples: Max number of samples to process (optional)
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Processing {len(data)} samples...")

    results = []
    for idx, sample in enumerate(data):
        video_path = Path(video_folder) / sample['video']

        # Extract question from conversations
        question = None
        for conv in sample.get('conversations', []):
            if conv['from'] == 'human':
                question = conv['value'].replace('<video>\n', '').strip()
                break

        if not question:
            question = "Describe what is happening in this video."

        print(f"\n[{idx+1}/{len(data)}] Processing {sample['id']}...")

        try:
            prediction = inference_video(
                processor, model, str(video_path), question
            )

            result = {
                'id': sample['id'],
                'video': sample['video'],
                'question': question,
                'prediction': prediction,
                'ground_truth': sample.get('conversations', [{}])[-1].get('value', '')
            }
            results.append(result)

        except Exception as e:
            print(f"❌ Error processing {sample['id']}: {e}")
            continue

    # Save results
    if output_path:
        print(f"\n💾 Saving results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✅ Results saved!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference with Qwen2-VL LoRA model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to checkpoint with LoRA weights (e.g., checkpoints/zac_qwen2vl_lora/checkpoint-latest)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to single video file for inference"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe what is happening in this video.",
        help="Question to ask about the video"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset JSON for batch inference"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        help="Folder containing videos for batch inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # Load model
    processor, model = load_model_with_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path
    )

    # Single video inference
    if args.video:
        inference_video(processor, model, args.video, args.question)

    # Batch inference
    elif args.data_path and args.video_folder:
        batch_inference(
            processor, model,
            args.data_path, args.video_folder,
            args.output, args.max_samples
        )

    else:
        print("❌ Please provide either --video or (--data_path and --video_folder)")
        parser.print_help()


if __name__ == "__main__":
    main()
