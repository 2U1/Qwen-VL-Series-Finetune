from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLPreTrainedModel
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.modeling_outputs import BaseModelOutputWithPooling

def replace_qwen2_5_vision():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel = Qwen2_5_VisionTransformerPretrainedModelWithPatchedWindow

class Qwen2_5_VisionTransformerPretrainedModelWithPatchedWindow(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)

            pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size
            pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size

            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens


    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> tuple | BaseModelOutputWithPooling:
        hidden_states = self.patch_embed(hidden_states)
        seq_len, dim = hidden_states.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens_list = self.get_window_index(grid_thw)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens_list,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        group = self.spatial_merge_unit
        G = seq_len // group

        hidden_states = hidden_states.view(G, group, dim)
        rotary_pos_emb = rotary_pos_emb.view(G, group, -1)

        window_index_dev = window_index.to(hidden_states.device, non_blocking=True)

        hidden_states = hidden_states.index_select(0, window_index_dev).reshape(seq_len, dim)
        rotary_pos_emb = rotary_pos_emb.index_select(0, window_index_dev).reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        if cu_seqlens.device != hidden_states.device:
             cu_seqlens = cu_seqlens.to(hidden_states.device, non_blocking=True)
             
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

        merged_hidden_states = self.merger(hidden_states)

        reverse_indices = torch.empty_like(window_index_dev)
        reverse_indices.scatter_(0, window_index_dev, torch.arange(window_index_dev.numel(), dtype=torch.long, device=window_index_dev.device))
        merged_hidden_states = merged_hidden_states.index_select(0, reverse_indices)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )
