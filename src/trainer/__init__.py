from importlib import import_module

__all__ = ["QwenSFTTrainer", "QwenDPOTrainer", "QwenGRPOTrainer", "QwenCLSTrainer", "GenerativeEvalPrediction"]


def __getattr__(name):
    if name in {"QwenSFTTrainer", "GenerativeEvalPrediction"}:
        module = import_module(".sft_trainer", __name__)
        return getattr(module, name)
    if name == "QwenDPOTrainer":
        module = import_module(".dpo_trainer", __name__)
        return getattr(module, name)
    if name == "QwenGRPOTrainer":
        module = import_module(".grpo_trainer", __name__)
        return getattr(module, name)
    if name == "QwenCLSTrainer":
        module = import_module(".cls_trainer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
