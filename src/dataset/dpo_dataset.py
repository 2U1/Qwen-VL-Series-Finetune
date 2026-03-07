import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import (
    chat_template_uses_reasoning_prefill,
    format_assistant_response,
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_video_info,
    model_supports_optional_reasoning,
    pad_sequence,
    replace_image_tokens,
    use_default_system_message,
)


class DPODataset(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(DPODataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )
        self.reasoning_supported = chat_template_uses_reasoning_prefill(self.processor, self.model_type)
        self.use_reasoning_prefill = self.data_args.enable_reasoning and self.reasoning_supported
        self.optional_reasoning_supported = model_supports_optional_reasoning(self.model_type)
        if self.data_args.enable_reasoning and not self.reasoning_supported:
            raise ValueError(
                f"`enable_reasoning` is only supported for Qwen3-VL Thinking or Qwen3.5 models with official reasoning chat templates. "
                f"Current model_type={self.model_type!r} does not qualify."
            )

    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        processor = self.processor

        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        all_input_ids = [] 
        all_prompt_mm_token_type_ids = []
        all_rejected = []
        all_chosen =[]
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids'] 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_prompt_mm_token_type_ids.append(torch.zeros_like(system_message_input_ids, dtype=torch.long).squeeze(0))

        user_prompt = replace_image_tokens(sources["prompt"], is_video=is_video)
        chosen_response = sources["chosen"]
        rejected_response = sources["rejected"]
        chosen_reasoning = sources.get("chosen_reasoning")
        rejected_reasoning = sources.get("rejected_reasoning")
        chosen_has_reasoning = isinstance(chosen_reasoning, str) and chosen_reasoning.strip()
        rejected_has_reasoning = isinstance(rejected_reasoning, str) and rejected_reasoning.strip()
        if self.data_args.enable_reasoning and chosen_has_reasoning != rejected_has_reasoning:
            raise ValueError(
                "Each DPO sample must provide both `chosen_reasoning` and `rejected_reasoning`, or neither of them."
            )
        if (
            self.data_args.enable_reasoning
            and self.reasoning_supported
            and not self.optional_reasoning_supported
            and not chosen_has_reasoning
        ):
            raise ValueError(
                "With `--enable_reasoning True`, Qwen3-VL Thinking DPO samples must include both `chosen_reasoning` and `rejected_reasoning`. "
                "Reasoning-optional samples are only supported for Qwen3.5."
            )
        use_reasoning_prefill = (
            self.use_reasoning_prefill
            and chosen_has_reasoning
            and rejected_has_reasoning
        )
        use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill
        assistant_prefill, chosen_response = format_assistant_response(
            chosen_response,
            chosen_reasoning,
            enable_reasoning=self.data_args.enable_reasoning,
            use_reasoning_prefill=use_reasoning_prefill,
            use_closed_think_prefill=use_closed_think_prefill,
        )
        _, rejected_response = format_assistant_response(
            rejected_response,
            rejected_reasoning,
            enable_reasoning=self.data_args.enable_reasoning,
            use_reasoning_prefill=use_reasoning_prefill,
            use_closed_think_prefill=use_closed_think_prefill,
        )

        user_input = (
            f"{DEFAULT_IM_START_TOKEN}user\n{user_prompt}{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}assistant\n{assistant_prefill}"
        )
        chosen_response = f"{chosen_response}{DEFAULT_IM_END_TOKEN}\n"
        rejected_response = f"{rejected_response}{DEFAULT_IM_END_TOKEN}\n"

        if DEFAULT_IMAGE_TOKEN in user_input:
            inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
            prompt_input_ids = inputs['input_ids']
            prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])
        elif DEFAULT_VIDEO_TOKEN in user_input:
            if self.model_type == "qwen2_5_vl":
                inputs = processor(
                    text=[user_input], 
                    images=images, 
                    videos=videos, 
                    padding=False, 
                    do_resize=False, 
                    return_tensors='pt', 
                    **video_kwargs
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                
                all_second_gird.extend(inputs["second_per_grid_ts"])
            
            elif self.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:

                video_datas, video_metadatas = zip(*videos)
                video_datas, video_metadatas = list(video_datas), list(video_metadatas)
                
                inputs = processor(
                    text=[user_input], 
                    images=images, 
                    videos=video_datas, 
                    padding=False, 
                    do_resize=False, 
                    return_tensors='pt', 
                    **video_kwargs, 
                    video_metadata=video_metadatas,
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
            
            else:
                inputs = processor(
                    text=[user_input], 
                    images=images, 
                    videos=videos, 
                    padding=False, 
                    do_resize=False, 
                    return_tensors='pt'
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
            
            prompt_input_ids = inputs['input_ids']
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])

        else:
            prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

        input_ids = prompt_input_ids.squeeze(0)
        chosen_input_ids = processor.tokenizer(chosen_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        rejected_input_ids = processor.tokenizer(rejected_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)

        all_input_ids.append(input_ids)
        all_prompt_mm_token_type_ids.append(prompt_mm_token_type_ids.squeeze(0))
        all_chosen.append(chosen_input_ids)
        all_rejected.append(rejected_input_ids)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        prompt_mm_token_type_ids = torch.cat(all_prompt_mm_token_type_ids, dim=0).to(torch.long)
        chosen = torch.cat(all_chosen, dim=0).to(torch.long)
        rejected = torch.cat(all_rejected, dim=0).to(torch.long)
        
        data_dict = dict(
            prompt_input_ids=input_ids,
            prompt_mm_token_type_ids=prompt_mm_token_type_ids,
            chosen_input_ids=chosen,
            rejected_input_ids=rejected,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
        
        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird

        return data_dict
    
class DataCollatorForDPODataset(object):
    """Collate examples for DPO fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_chosen_ids = []
        batch_rejected_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_prompt_mm_token_type_ids = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["prompt_input_ids"])
            batch_prompt_mm_token_type_ids.append(example["prompt_mm_token_type_ids"])
            batch_chosen_ids.append(example["chosen_input_ids"])
            batch_rejected_ids.append(example["rejected_input_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        prompt_input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        prompt_mm_token_type_ids = pad_sequence(
            batch_prompt_mm_token_type_ids, padding_side='right', padding_value=0
        )

        chosen = pad_sequence(batch_chosen_ids, padding_side='right', padding_value=self.pad_token_id)
        rejected = pad_sequence(batch_rejected_ids, padding_side='right', padding_value=self.pad_token_id)

        # torch.argmax used in `trl.trainer.utils.flush_left` does not accept bool tensors on some torch versions
        # (e.g., torch==2.1); keep masks int to stay compatible.
        prompt_attention_mask = (prompt_input_ids != self.pad_token_id).long()
        chosen_attention_mask = (chosen != self.pad_token_id).long()
        rejected_attention_mask = (rejected != self.pad_token_id).long()


        data_dict = {
            'prompt_input_ids': prompt_input_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'prompt_mm_token_type_ids': prompt_mm_token_type_ids,
            'chosen_input_ids': chosen,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected,
            'rejected_attention_mask': rejected_attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    
def make_dpo_data_module(model_id, processor, data_args):
    """Make dataset and collator for DPO fine-tuning."""
    dpo_dataset = DPODataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForDPODataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=dpo_dataset,
                eval_dataset=None,
                data_collator=data_collator)
