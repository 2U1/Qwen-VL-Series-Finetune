import os
import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from typing import Union

import trl.import_utils as trl_import_utils
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)


def _normalize_trl_optional_flags():
    for name in dir(trl_import_utils):
        if not (name.startswith("_") and name.endswith("_available")):
            continue
        value = getattr(trl_import_utils, name)
        if isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])


_normalize_trl_optional_flags()

from trl import DPOTrainer
from trl.trainer.utils import pad_to_length, flush_left, selective_log_softmax
from train.train_utils import get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenDPOTrainer(DPOTrainer):

    def __init__(self, *args, **kwargs):
        super(QwenDPOTrainer, self).__init__(*args, **kwargs)

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        dataset_name
    ):
        return dataset

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:

        concatenated_batch = {}

        concatenated_batch['prompt_input_ids'] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        concatenated_batch['prompt_attention_mask'] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)
        if "prompt_mm_token_type_ids" in batch:
            concatenated_batch["prompt_mm_token_type_ids"] = torch.cat(
                [batch["prompt_mm_token_type_ids"], batch["prompt_mm_token_type_ids"]], dim=0
            )

        if 'pixel_values' in batch:
            concatenated_batch['pixel_values'] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
            concatenated_batch['image_grid_thw'] = torch.cat([batch["image_grid_thw"], batch["image_grid_thw"]], dim=0)

        if 'pixel_values_videos' in batch:
            concatenated_batch['pixel_values_videos'] = torch.cat(
                [batch["pixel_values_videos"], batch["pixel_values_videos"]], dim=0
            )
            concatenated_batch['video_grid_thw'] = torch.cat(
                [batch["video_grid_thw"], batch["video_grid_thw"]], dim=0
            )

        if "second_per_grid_ts" in batch:
            concatenated_batch["second_per_grid_ts"] = batch["second_per_grid_ts"] + batch["second_per_grid_ts"]

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        concatenated_batch['completion_input_ids'] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )

        concatenated_batch['completion_attention_mask'] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return concatenated_batch
    

    def concatenated_forward(self, model, batch, is_ref_model:bool=False):

        num_examples = batch['prompt_input_ids'].shape[0]
        
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}

        if self.aux_loss_enabled:
            model_kwargs['output_router_logits'] = True

        # Add image/video values to model kwargs
        if 'pixel_values' in batch:
            model_kwargs['pixel_values'] = concatenated_batch['pixel_values']
            model_kwargs['image_grid_thw'] = concatenated_batch['image_grid_thw']
        if 'pixel_values_videos' in batch:
            model_kwargs['pixel_values_videos'] = concatenated_batch['pixel_values_videos']
            model_kwargs['video_grid_thw'] = concatenated_batch['video_grid_thw']
        if "second_per_grid_ts" in batch:
            model_kwargs["second_per_grid_ts"] = concatenated_batch["second_per_grid_ts"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        prompt_mm_token_type_ids = concatenated_batch.get("prompt_mm_token_type_ids")
        
        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Flush left to reduce the memory usage
        # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
        #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
        if prompt_mm_token_type_ids is not None:
            completion_mm_token_type_ids = torch.zeros_like(completion_input_ids)
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, completion_mm_token_type_ids), dim=1)
            attention_mask, input_ids, loss_mask, mm_token_type_ids = flush_left(
                attention_mask, input_ids, loss_mask, mm_token_type_ids
            )
            model_kwargs["mm_token_type_ids"] = mm_token_type_ids
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        model_kwargs["attention_mask"] = attention_mask

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_decay"
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_non_decay"
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_decay",
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_non_decay",
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )

        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))
