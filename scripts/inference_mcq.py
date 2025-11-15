#!/usr/bin/env python3
"""
Inference script for Multiple Choice Questions (MCQ) with Qwen2-VL
Specifically designed for ZAC2025 traffic dataset format
"""

import torch
import argparse
import sys
import re
from pathlib import Path
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json


def load_model_with_lora(
    base_model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: str = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16
):
    """Load Qwen2-VL model with LoRA adapter and merger weights"""
    print(f"Loading base model from {base_model_path}...")
    print(f"Target device: {device}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    if lora_path:
        lora_path = Path(lora_path)

        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(
            model,
            str(lora_path),
            is_trainable=False
        )

        # Load merger weights
        merger_weights_path = lora_path / "merger_weights.bin"
        if merger_weights_path.exists():
            print(f"Loading merger weights from {merger_weights_path}...")
            merger_weights = torch.load(merger_weights_path, map_location="cpu")

            model_state = model.state_dict()
            for name, param in merger_weights.items():
                clean_name = name.replace('base_model.model.', '')
                if clean_name in model_state:
                    model_state[clean_name].copy_(param)
                    print(f"  Loaded: {clean_name}")

            print(f"Successfully loaded {len(merger_weights)} merger parameters")

    model.eval()
    print("✅ Model loaded successfully!")
    return processor, model


def extract_answer(response: str) -> str:
    """
    Extract answer letter (A/B/C/D) from model response

    Tries multiple patterns:
    1. "Đáp án: X"
    2. Last "X. " pattern
    3. Standalone A/B/C/D at end
    """
    # Method 1: "Đáp án: X"
    match = re.search(r'Đáp án:\s*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Method 2: Last "X. " pattern in text
    matches = re.findall(r'\b([A-D])\.\s+', response)
    if matches:
        return matches[-1].upper()

    # Method 3: Standalone letter at end
    match = re.search(r'\b([A-D])\s*$', response.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def inference_mcq(
    processor,
    model,
    video_path: str,
    question: str,
    max_new_tokens: int = 512
):
    """
    Run inference on a multiple choice question

    Returns:
        dict: {
            'full_response': str,  # Full model output
            'answer': str,         # Extracted answer (A/B/C/D)
            'reasoning': str       # Response without answer line
        }
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

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

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    full_response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Extract answer
    answer = extract_answer(full_response)

    # Extract reasoning (remove answer line)
    reasoning = re.sub(r'\d+\.\s*Đáp án:.*', '', full_response).strip()

    return {
        'full_response': full_response,
        'answer': answer,
        'reasoning': reasoning
    }


def batch_evaluate_mcq(
    processor,
    model,
    data_path: str,
    video_folder: str,
    output_path: str = None,
    max_samples: int = None
):
    """
    Evaluate model on MCQ dataset

    Returns accuracy and detailed results
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Evaluating on {len(data)} samples...")

    results = []
    correct = 0
    total = 0

    for idx, sample in enumerate(data):
        video_path = Path(video_folder) / sample['video']

        # Extract question and ground truth
        question = None
        ground_truth_text = None
        for conv in sample.get('conversations', []):
            if conv['from'] == 'human':
                question = conv['value'].replace('<video>\n', '').strip()
            elif conv['from'] == 'gpt':
                ground_truth_text = conv['value']

        if not question:
            print(f"⚠️  Skipping {sample['id']}: No question found")
            continue

        # Extract ground truth answer
        true_answer = extract_answer(ground_truth_text) if ground_truth_text else None

        print(f"\n[{idx+1}/{len(data)}] {sample['id']}")

        try:
            result_dict = inference_mcq(
                processor, model, str(video_path), question
            )

            pred_answer = result_dict['answer']
            is_correct = (pred_answer == true_answer) if true_answer else None

            if is_correct:
                correct += 1
            if true_answer:
                total += 1

            result = {
                'id': sample['id'],
                'video': sample['video'],
                'question': question,
                'full_response': result_dict['full_response'],
                'predicted_answer': pred_answer,
                'ground_truth_answer': true_answer,
                'correct': is_correct
            }
            results.append(result)

            status = '✅' if is_correct else '❌' if is_correct is not None else '⚪'
            print(f"  Predicted: {pred_answer} | Truth: {true_answer} | {status}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"Successfully processed: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")

    # Save results
    if output_path:
        output_data = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }

        print(f"\n💾 Saving results to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print("✅ Results saved!")

    return results, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="MCQ inference with Qwen2-VL LoRA model"
    )
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
        help="Path to checkpoint with LoRA weights"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to single video file"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="MCQ question with choices"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset JSON"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        help="Folder containing videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max samples to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on"
    )

    args = parser.parse_args()

    # Load model
    processor, model = load_model_with_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device
    )

    # Single video inference
    if args.video and args.question:
        result = inference_mcq(processor, model, args.video, args.question)
        print(f"\n{'='*60}")
        print(f"INFERENCE RESULT")
        print(f"{'='*60}")
        print(f"Video: {args.video}")
        print(f"Question: {args.question[:100]}...")
        print(f"\nFull Response:\n{result['full_response']}")
        print(f"\nExtracted Answer: {result['answer']}")
        print(f"{'='*60}")

    # Batch evaluation
    elif args.data_path and args.video_folder:
        batch_evaluate_mcq(
            processor, model,
            args.data_path, args.video_folder,
            args.output, args.max_samples
        )

    else:
        print("❌ Provide either (--video and --question) or (--data_path and --video_folder)")
        parser.print_help()


if __name__ == "__main__":
    main()
