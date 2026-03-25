"""
Prepare sample datasets for Qwen-VL fine-tuning.

Supported tasks:
- sft: TextVQA samples in the conversational SFT format.
- classification: generic image classification samples in the sequence-classification format.

Usage:
    python prepare_sample_data.py
    python prepare_sample_data.py --task classification --num_samples 50
"""
import argparse
from collections import Counter
import json
import os

from datasets import ClassLabel, load_dataset


DEFAULT_CLASSIFICATION_PROMPT = "Question: What is the correct class for this image?\nOptions:\n{options}"


def save_rgb_image(image, output_path):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(output_path)


def infer_label_names(dataset, label_column):
    features = getattr(dataset, "features", None)
    if not features or label_column not in features:
        return None

    feature = features[label_column]
    if isinstance(feature, ClassLabel):
        return list(feature.names)

    names = getattr(feature, "names", None)
    if names:
        return list(names)

    return None


def format_classification_prompt(label_names):
    options = [f"{idx}. {label_name}" for idx, label_name in enumerate(label_names)]
    return DEFAULT_CLASSIFICATION_PROMPT.format(options="\n".join(options))


def prepare_sft_samples(args, image_dir):
    print(f"Downloading TextVQA dataset (streaming, {args.num_samples} samples)...")
    dataset = load_dataset("textvqa", split=args.split, streaming=True)

    samples = []
    for idx, item in enumerate(dataset):
        if idx >= args.num_samples:
            break

        image_filename = f"{idx:04d}.jpg"
        save_rgb_image(item["image"], os.path.join(image_dir, image_filename))

        answer = item["answers"][0] if item["answers"] else "unknown"
        sample = {
            "image": image_filename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": answer},
            ],
        }
        samples.append(sample)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{args.num_samples} done...")

    dataset_path = os.path.join(args.output_dir, "train.json")
    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(samples, file, indent=2, ensure_ascii=False)

    return dataset_path, None, None


def prepare_classification_samples(args, image_dir):
    dataset_name = args.classification_dataset
    print(
        f"Downloading {dataset_name} dataset (streaming, {args.num_samples} samples)..."
    )
    dataset = load_dataset(dataset_name, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    sample_item = next(iter(dataset))
    if "image" in sample_item:
        image_column = "image"
    elif "img" in sample_item:
        image_column = "img"
    else:
        raise ValueError(
            f"Could not find an image column in dataset {dataset_name}. "
            f"Available keys: {sorted(sample_item.keys())}"
        )

    if "label" in sample_item:
        label_column = "label"
    elif "labels" in sample_item:
        label_column = "labels"
    else:
        raise ValueError(
            f"Could not find a label column in dataset {dataset_name}. "
            f"Available keys: {sorted(sample_item.keys())}"
        )

    label_names = infer_label_names(dataset, label_column)
    if label_names is None:
        raise ValueError(
            f"Could not infer class names from dataset {dataset_name}. "
            f"Please use a dataset with a ClassLabel label feature on `{label_column}`."
        )

    prompt = format_classification_prompt(label_names)
    label_map_path = os.path.join(args.output_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as file:
        json.dump(
            {label_name: idx for idx, label_name in enumerate(label_names)},
            file,
            indent=2,
            ensure_ascii=False,
        )

    dataset = load_dataset(dataset_name, split=args.split, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    samples = []
    saved_label_counter = Counter()
    seen_labels = set()
    saved_count = 0
    ensure_class_coverage = args.ensure_class_coverage and args.num_samples >= len(label_names)

    for item in dataset:
        if saved_count >= args.num_samples:
            break

        label_idx = int(item[label_column])
        label_name = label_names[label_idx]

        if ensure_class_coverage and len(seen_labels) < len(label_names) and label_name in seen_labels:
            continue

        image_filename = f"{saved_count:04d}.jpg"
        save_rgb_image(item[image_column], os.path.join(image_dir, image_filename))

        sample = {
            "id": str(item.get("id", saved_count)),
            "image": image_filename,
            "prompt": prompt,
            "label": label_name,
        }
        samples.append(sample)
        saved_label_counter[sample["label"]] += 1
        seen_labels.add(label_name)
        saved_count += 1

        if saved_count % 20 == 0:
            print(f"  {saved_count}/{args.num_samples} done...")

    if ensure_class_coverage and len(seen_labels) < len(label_names):
        print(
            "Warning: could not cover all classes in the saved samples. "
            f"Covered {len(seen_labels)}/{len(label_names)} classes."
        )

    dataset_path = os.path.join(args.output_dir, "train_cls.json")
    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(samples, file, indent=2, ensure_ascii=False)

    return dataset_path, label_map_path, label_names, saved_label_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["sft", "classification"], default="sft")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--classification_dataset", type=str, default="beans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000)
    parser.add_argument("--disable_class_coverage", action="store_true")
    args = parser.parse_args()
    args.ensure_class_coverage = not args.disable_class_coverage

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    if args.task == "classification":
        dataset_path, label_map_path, class_names, saved_label_counter = prepare_classification_samples(args, image_dir)
    else:
        dataset_path, label_map_path, class_names = prepare_sft_samples(args, image_dir)
        saved_label_counter = None

    print(f"\nDone! Samples saved.")
    print(f"  Task: {args.task}")
    print(f"  Images: {image_dir}/")
    print(f"  Dataset: {dataset_path}")
    if label_map_path is not None:
        print(f"  Label map: {label_map_path}")
    if class_names is not None:
        print(f"  Num classes: {len(class_names)}")
        print(f"  Classes: {class_names}")
    if saved_label_counter is not None:
        print(f"  Saved classes: {sorted(saved_label_counter)}")
        print(f"  Saved class counts: {dict(saved_label_counter)}")
    print(f"\nUsage in training script:")
    print(f"  --data_path {dataset_path}")
    print(f"  --image_folder {image_dir}")


if __name__ == "__main__":
    main()
