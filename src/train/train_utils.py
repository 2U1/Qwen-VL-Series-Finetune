import transformers
import torch
import logging
import inspect


_ZERO3_MODULE_LOCAL_ATTRIBUTES = frozenset({
    "pre_bwd_fn",
    "post_bwd_fn",
    "applied_pre_backward_ref_cnt",
})


def _patch_peft_auxiliary_wrapper_for_zero3(wrapper_cls):
    """Prevent PEFT wrappers from delegating DeepSpeed's module-local hook state."""
    original_getattr = wrapper_cls.__getattr__
    if getattr(original_getattr, "_qwen_zero3_safe", False):
        return False

    def zero3_safe_getattr(self, name):
        if name in _ZERO3_MODULE_LOCAL_ATTRIBUTES:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return original_getattr(self, name)

    zero3_safe_getattr._qwen_zero3_safe = True
    wrapper_cls.__getattr__ = zero3_safe_getattr
    return True


def patch_deepspeed_zero3_peft_hooks():
    """Backport the ZeRO-3 fix for PEFT's attribute-delegating wrappers.

    Released DeepSpeed versions through 0.19.2 use ``hasattr`` when creating
    module-local backward-hook state. PEFT's ``ModulesToSaveWrapper`` delegates
    missing attributes to its child, which can make the parent reuse the child's
    hook closures and fail with a missing ``ds_grads_remaining`` attribute.

    DeepSpeed fixed this upstream by consulting ``module.__dict__``. Until that
    fix is in a release, stop PEFT from delegating only those private hook-state
    attributes. The workaround becomes a no-op once the upstream implementation
    is detected.
    """
    try:
        from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
        from peft.utils.other import AuxiliaryTrainingWrapper
    except (ImportError, AttributeError):
        return False

    try:
        register_source = inspect.getsource(
            DeepSpeedZeRoOffload._register_deepspeed_module
        )
    except (OSError, TypeError):
        return False

    if '"pre_bwd_fn" not in module.__dict__' in register_source:
        return False
    if 'hasattr(module, "pre_bwd_fn")' not in register_source:
        return False

    return _patch_peft_auxiliary_wrapper_for_zero3(AuxiliaryTrainingWrapper)


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    named_params = list(named_params)
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for bias_name, tensor in maybe_lora_bias.items():
            if bias_name in lora_bias_names:
                to_return[bias_name] = tensor
    else:
        raise NotImplementedError

    # PEFT stores trainable classification heads (and similar auxiliary
    # modules) under modules_to_save. Keep them in the adapter state alongside
    # LoRA tensors so save_pretrained can normalize their key names.
    to_return.update({
        k: t for k, t in named_params if "modules_to_save" in k
    })
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {
        k: t
        for k, t in named_params
        if "lora_" not in k and "modules_to_save" not in k
    }
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)
