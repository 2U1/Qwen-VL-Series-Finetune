from pathlib import Path
from peft import PeftModel
import torch
from transformers import (
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    AutoConfig, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
)
import warnings
import os
import json
import importlib
import inspect
from types import ModuleType
from typing import Callable, List

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if is_lora_model(model_path) and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if is_lora_model(model_path) and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base)
        print('Loading Qwen2-VL from base model...')
        if "Qwen3" in model_base:
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        elif "Qwen2.5" in model_base:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Qwen2-VL weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        print(f"Loading model from {model_path} as a standard model. Adapter files were not found, so it can't be merged")
        config_path = Path(model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        processor = AutoProcessor.from_pretrained(model_path)
        
        architecture = config.get("architectures", [None])[0]
        if "Qwen3" in architecture:
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif "Qwen2_5" in architecture:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model

def is_lora_model(model_path: str | Path) -> bool:
    """
    Check if a model directory contains LoRA adapter files.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        bool: True if the directory contains LoRA adapter files
    """
    model_dir = Path(model_path)
    return (model_dir / 'adapter_config.json').exists() and (model_dir / 'adapter_model.safetensors').exists()

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
def load_reward_funcs(
    module_path: str = "train.reward_funcs",
    *,
    name_pred = lambda n: n.endswith("_reward"),
    obj_pred  = lambda o: callable(o),
    keep_order: bool = True
) -> List[Callable]:

    mod: ModuleType = importlib.import_module(module_path)
    
    members = inspect.getmembers(mod, predicate=obj_pred)

    reward_funcs = [(n, o) for n, o in members if name_pred(n)]

    if keep_order:
        reward_funcs.sort(key=lambda pair: inspect.getsourcelines(pair[1])[1])

    return [o for _, o in reward_funcs]