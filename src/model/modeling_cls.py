import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union, Dict, Any, List

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLPreTrainedModel,
    Qwen2VLModel,
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from train.monkey_patch_vision import replace_qwen2_5_vision

replace_qwen2_5_vision()

class Qwen2VLForSequenceClassification(Qwen2VLPreTrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        bridge_h = config.mlp_head_hidden_dim
        bridge_p = config.mlp_head_dropout

        self.model = Qwen2VLModel(config)
        
        self.bridge = None
        in_dim = config.hidden_size
        if bridge_h > 0:
            self.bridge = nn.Sequential(
                nn.Linear(config.hidden_size, bridge_h),
                nn.GELU(),
                nn.Dropout(bridge_p),
            )
            nn.init.xavier_uniform_(self.bridge[0].weight, gain=1.0)
            nn.init.zeros_(self.bridge[0].bias)
            in_dim = bridge_h
            
        self.score = nn.Linear(in_dim, self.num_labels, bias=False)
        nn.init.normal_(self.score.weight, std=1e-3)

        self.loss_fn = None
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()
    
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)
    
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        feats = self.bridge(hidden_states) if self.bridge is not None else hidden_states

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        if self.config.pad_token_id is None:
            sequence_lengths = torch.full((batch_size,), -1, device=feats.device)
        else:
            if input_ids is not None:
                non_pad_mask = (input_ids != self.config.pad_token_id).to(feats.device)

                token_indices = torch.arange(
                    input_ids.size(-1), device=feats.device, dtype=torch.long
                )
                sequence_lengths = (token_indices * non_pad_mask).argmax(dim=-1)
            else:
                sequence_lengths = torch.full((batch_size,), -1, device=feats.device)

        pooled_feats = feats[torch.arange(batch_size, device=feats.device), sequence_lengths]
        pooled_logits = self.score(pooled_feats)

        loss: Optional[torch.Tensor] = None

        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                # automatically infer
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (
                    torch.long,
                    torch.int,
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                if hasattr(self, "loss_fn") and self.loss_fn is not None:
                    loss_fct = self.loss_fn
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            else:  # multi-label
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class Qwen2_5_VLForSequenceClassification(Qwen2_5_VLPreTrainedModel):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    accepts_loss_kwargs = False
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        bridge_h = config.mlp_head_hidden_dim
        bridge_p = config.mlp_head_dropout
        self.model = Qwen2_5_VLModel(config)
        
        self.bridge = None
        in_dim = config.hidden_size
        if bridge_h > 0:
            self.bridge = nn.Sequential(
                nn.Linear(config.hidden_size, bridge_h),
                nn.GELU(),
                nn.Dropout(bridge_p),
            )
            nn.init.xavier_uniform_(self.bridge[0].weight, gain=1.0)
            nn.init.zeros_(self.bridge[0].bias)
            in_dim = bridge_h
            
        self.score = nn.Linear(in_dim, self.num_labels, bias=False)
        nn.init.normal_(self.score.weight, std=1e-3)

        self.loss_fn = None
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()
    
    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        feats = self.bridge(hidden_states) if self.bridge is not None else hidden_states


        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        if self.config.pad_token_id is None:
            sequence_lengths = torch.full((batch_size,), -1, device=feats.device)
        else:
            if input_ids is not None:
                non_pad_mask = (input_ids != self.config.pad_token_id).to(feats.device)

                token_indices = torch.arange(
                    input_ids.size(-1), device=feats.device, dtype=torch.long
                )
                sequence_lengths = (token_indices * non_pad_mask).argmax(dim=-1)
            else:
                sequence_lengths = torch.full((batch_size,), -1, device=feats.device)
        
        
        pooled_feats = feats[torch.arange(batch_size, device=feats.device), sequence_lengths]
        pooled_logits = self.score(pooled_feats)

        loss: Optional[torch.Tensor] = None
        
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                # automatically infer
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (
                    torch.long,
                    torch.int,
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                if hasattr(self, "loss_fn") and self.loss_fn is not None:
                    loss_fct = self.loss_fn
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            else:  # multi-label
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
