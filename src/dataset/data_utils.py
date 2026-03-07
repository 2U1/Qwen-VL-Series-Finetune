import re
import torch
from functools import lru_cache

from transformers import AutoConfig

from qwen_vl_utils import process_vision_info

from constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n*' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n*'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n*' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n*'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        if "reasoning" in conversation:
            transformed_entry["reasoning"] = conversation["reasoning"]
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_mm_token_type_ids(inputs, input_ids):
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        return torch.zeros_like(input_ids, dtype=torch.long)
    return mm_token_type_ids.to(dtype=torch.long)


@lru_cache(maxsize=32)
def get_qwen_multimodal_settings(model_id_or_path):
    model_type = AutoConfig.from_pretrained(model_id_or_path).model_type
    if model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
        return model_type, 16, True
    return model_type, 14, False


def use_default_system_message(model_type):
    return model_type in {"qwen2_vl", "qwen2_5_vl"}


def chat_template_uses_reasoning_prefill(processor, model_type=None):
    template = getattr(processor, "chat_template", None)
    if not template and hasattr(processor, "tokenizer"):
        template = getattr(processor.tokenizer, "chat_template", None)
    template = template or ""
    supported_model_types = {"qwen3_vl", "qwen3_5", "qwen3_5_moe"}
    if model_type not in supported_model_types:
        return False
    return (
        "reasoning_content" in template
        and "<think>" in template
        and "add_generation_prompt" in template
        and "<|im_start|>assistant" in template
    )


def model_supports_optional_reasoning(model_type):
    return model_type in {"qwen3_5", "qwen3_5_moe"}


def format_assistant_response(
    content,
    reasoning=None,
    *,
    enable_reasoning=False,
    use_reasoning_prefill=False,
    use_closed_think_prefill=False,
):
    if use_closed_think_prefill:
        return "<think>\n\n</think>\n\n", content.lstrip("\n")

    if not enable_reasoning or not isinstance(reasoning, str) or not reasoning.strip():
        return "", content

    reasoning = reasoning.strip("\n")
    content = content.lstrip("\n")

    if use_reasoning_prefill:
        return "<think>\n", f"{reasoning}\n</think>\n\n{content}"

    return "", f"<think>\n{reasoning}\n</think>\n\n{content}"

def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps, image_patch_size, return_video_metadata=False):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(
        messages, 
        return_video_kwargs=True, 
        image_patch_size=image_patch_size, 
        return_video_metadata=return_video_metadata
    )

    return video_input[0], video_kwargs

def samples_per_class_from_ids(label_ids, num_classes):
    
    counts = torch.bincount(
        torch.as_tensor(label_ids, dtype=torch.long),
        minlength=num_classes
    )
    
    return counts.tolist()
