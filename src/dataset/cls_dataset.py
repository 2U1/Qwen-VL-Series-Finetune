import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

from params import DataArguments
from constants import (
    SYSTEM_MESSAGE,
)

from .data_utils import (
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    pad_sequence,
    samples_per_class_from_ids,
    use_default_system_message,
)

CLASS_2_ID = {
    "A": 0,
    "B": 1
}

USER_MESSAGE = """Enter your prompt here. This will be used when your data does not have a prompt field."""

def get_image_content(image_path, min_pixel, max_pixel, width, height):
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content

def get_video_content(video_path, min_pixels, max_pixels, width, height, fps, nframes):
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

    if nframes is not None:
        content["nframes"] = nframes
    else:
        content["fps"] = fps

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    return content

class ClassificationDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(ClassificationDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.compute_dtype = data_args.compute_dtype

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
        self.model_type, _, _ = get_qwen_multimodal_settings(self.model_id)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        contents = []
        
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_image_content(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            frame_paths = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                    frame_paths.append(video_file)
            
            contents.append(get_video_content(frame_paths, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.fps, self.nframes))

        if "prompt" in sources:
            text_content = {"type": "text", "text": sources["prompt"]}

        else: 
            text_content = {"type": "text", "text": USER_MESSAGE}
        
        contents.append(text_content)

        user_prompt = [{"role": "user", "content": contents}]

        if len(SYSTEM_MESSAGE) > 0 and use_default_system_message(self.model_type):
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)

        text = self.processor.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(user_prompt, return_video_kwargs=True)
        video_kwargs = {
            key: value
            for key, value in video_kwargs.items()
            if value is not None and value != []
        }

        data_dict = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs
        )
        mm_token_type_ids = get_mm_token_type_ids(data_dict, data_dict["input_ids"])

        labels = [torch.tensor(CLASS_2_ID[sources["label"]], dtype=torch.long)]

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (data_dict['input_ids'] > -1000000).to(torch.long)

        data_dict['labels'] = labels
        data_dict['attention_mask'] = attention_mask
        data_dict['mm_token_type_ids'] = mm_token_type_ids

        for key, value in data_dict.items():  # cast data dtype for paligemma
            if torch.is_tensor(value) and torch.is_floating_point(value):
                data_dict[key] = value.to(self.compute_dtype)
        
        return data_dict

class DataCollatorForClassificationDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, padding_side: str = "right"):
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, examples):
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_mm_token_type_ids = []
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"].squeeze(0))
            batch_labels.extend(example["labels"])
            batch_mm_token_type_ids.append(example["mm_token_type_ids"].squeeze(0))

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side=self.padding_side, padding_value=self.pad_token_id
        )
        labels = torch.tensor(batch_labels, dtype=torch.long)
        mm_token_type_ids = pad_sequence(
            batch_mm_token_type_ids, padding_side=self.padding_side, padding_value=0
        )

        attention_mask = input_ids != self.pad_token_id

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'mm_token_type_ids': mm_token_type_ids,
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
    
def make_classification_data_module(model_id, processor, data_args):

    eval_ds = None
    eval_data_collator = None

    cls_dataset = ClassificationDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    train_data_collator = DataCollatorForClassificationDataset(pad_token_id=processor.tokenizer.pad_token_id, padding_side="left")

    labels_list = [CLASS_2_ID[s["label"]] for s in cls_dataset.list_data_dict]

    samples_per_class = samples_per_class_from_ids(
        labels_list, num_classes=len(CLASS_2_ID)
    )

    if data_args.eval_path is not None:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.image_folder = data_args.eval_image_folder
        eval_data_args.data_path = data_args.eval_path
        eval_ds = ClassificationDataset(
            data_path=eval_data_args.data_path, processor=processor, data_args=eval_data_args, model_id=model_id
        )
        eval_data_collator = DataCollatorForClassificationDataset(pad_token_id=processor.tokenizer.pad_token_id, padding_side="left")

    return dict(
            train_dataset=cls_dataset,
            eval_dataset=eval_ds,
            train_data_collator=train_data_collator,
            eval_data_collator=eval_data_collator,
            samples_per_class=samples_per_class,
        )
