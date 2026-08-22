import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from utils import get_model_name_from_path, load_pretrained_model


def _base_checkpoint_shards(base_path: Path):
    index_path = base_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        return sorted({base_path / name for name in weight_map.values()})
    return sorted(base_path.glob("model*.safetensors"))


def _to_model_key(key, conversion_mapping):
    """Rename a checkpoint key the way `from_pretrained` would."""
    for pattern, replacement in conversion_mapping.items():
        renamed = re.sub(pattern, replacement, key)
        if renamed != key:
            return renamed
    return key


def collect_base_only_tensors(model_base, model):
    """Tensors that exist in the base checkpoint but not in the instantiated model.

    Qwen3.5 ships a multi-token-prediction head (`mtp.*`, 15 tensors). It is listed in
    `Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected`, so it is dropped on load
    and never becomes a submodule. `save_pretrained` can only write what the module holds,
    which means a merged checkpoint loses those tensors permanently and stops being a
    drop-in replacement for the base model. Carry them over.

    `_checkpoint_conversion_mapping` has to be applied first. Qwen2-VL and Qwen2.5-VL keep
    their published checkpoints in the pre-refactor layout (`visual.*`, `model.*`) and
    transformers renames them on load, so a naive key comparison would classify the whole
    checkpoint as "missing from the model" and copy a second, unmerged, stale-layout copy
    of every weight into the output.
    """
    base_path = Path(model_base)
    if not base_path.is_dir():
        print(f"Note: '{model_base}' is not a local directory; skipping the base-only weight check.")
        return {}

    shards = _base_checkpoint_shards(base_path)
    if not shards:
        print(f"Note: no safetensors shards found under {base_path}; skipping the base-only weight check.")
        return {}

    model_keys = set(model.state_dict())
    conversion_mapping = getattr(type(model), "_checkpoint_conversion_mapping", None) or {}

    extra = {}
    total = 0
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += 1
                if _to_model_key(key, conversion_mapping) not in model_keys:
                    extra[key] = handle.get_tensor(key)

    # Belt and braces: a genuinely extra head is a handful of tensors. If a large slice of
    # the checkpoint looks unaccounted for, the key layouts do not line up for some reason
    # this function does not know about -- keep the old behaviour rather than write a
    # checkpoint with two copies of everything.
    if extra and len(extra) > total // 4:
        print(f"Warning: {len(extra)} of {total} base tensors have no counterpart in the model, "
              f"which does not look like an extra head. Skipping the base-only weight copy.")
        return {}

    return extra


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map='cpu')

    state_dict = model.state_dict()
    base_only = collect_base_only_tensors(args.model_base, model)
    if base_only:
        # The model is loaded in a fixed dtype, so the rest of the checkpoint may not be in
        # the base checkpoint's dtype any more. Match it, otherwise the output is a
        # mixed-dtype file that disagrees with its own config.
        target_dtype = next((t.dtype for t in state_dict.values() if t.is_floating_point()), None)
        if target_dtype is not None:
            base_only = {k: (v.to(target_dtype) if v.is_floating_point() else v)
                         for k, v in base_only.items()}
        prefixes = sorted({key.split('.', 1)[0] for key in base_only})
        print(f"Preserving {len(base_only)} tensor(s) that exist only in the base checkpoint "
              f"(top-level prefixes: {prefixes}).")
        state_dict.update(base_only)

    model.save_pretrained(args.save_model_path, state_dict=state_dict,
                          safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action='store_true')

    args = parser.parse_args()

    merge_lora(args)
