"""
Prepare a sample dataset for Qwen2-VL fine-tuning.
Downloads samples from HuggingFace TextVQA and saves them to the data/ folder.

Usage:
    python prepare_sample_data.py
    python prepare_sample_data.py --num_samples 50
"""
import os
import json
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data")
    args = parser.parse_args()

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    print(f"Downloading TextVQA dataset (streaming, {args.num_samples} samples)...")
    ds = load_dataset("textvqa", split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= args.num_samples:
            break

        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_filename = f"{i:04d}.jpg"
        img.save(os.path.join(image_dir, img_filename))

        answer = item["answers"][0] if item["answers"] else "unknown"

        sample = {
            "image": img_filename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": answer},
            ],
        }
        samples.append(sample)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.num_samples} done...")

    train_path = os.path.join(args.output_dir, "train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(samples)} samples saved.")
    print(f"  Images: {image_dir}/")
    print(f"  Dataset: {train_path}")
    print(f"\nUsage in training script:")
    print(f"  --data_path {train_path}")
    print(f"  --image_folder {image_dir}")


if __name__ == "__main__":
    main()
