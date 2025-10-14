from copy import deepcopy
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss, LayerNorm
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import warnings
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

import math
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from qwen25vl.modeling_qwen2_5_vl import *
from qwen25vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel, 
    Qwen2_5_VisionPatchEmbed, 
    Qwen2_5_VisionRotaryEmbedding, 
    Qwen2_5_VLVisionBlock, 
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionFlashAttention2,
    Qwen2RMSNorm,
    Qwen2_5_VLMLP,
    apply_rotary_pos_emb_flashatt
    )
from .selector_scorer import TransformerScorer
import torch.nn.functional as F
from torch import vmap
from torch.func import grad
from torch.autograd import Function
import os

# --- Start of new Differentiable TopK implementation ---
sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))

class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k):
        ts, ps = _find_ts(xs, k)
        ctx.save_for_backward(xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts = ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = sigmoid_grad(xs + ts)
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

@torch.no_grad()
def _find_ts(xs, k):
    b, n = xs.shape
    assert 0 < k < n
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid(xs + mid).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(xs + ts)

topk = TopK.apply
# --- End of new Differentiable TopK implementation ---



class Qwen2_5_VisionTransformerPretrainedModel_Selector(Qwen2_5_VisionTransformerPretrainedModel):
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
            [Qwen2_5_VLVisionBlock(config, config._attn_implementation, layer_idx=i) for i in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False
        self.importance_scorer = TransformerScorer(in_features=config.out_hidden_size,hidden_dim=config.out_hidden_size//2)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        len_blocks = len(self.blocks)
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

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        # ---------------------------add--------------------------------------------------------
        total_token_num = hidden_states.shape[0]
        hidden_states_unsqueezed = hidden_states.unsqueeze(0)
        learned_scores = self.importance_scorer(hidden_states_unsqueezed.detach()).squeeze(0) 
        dominant_num = max(1, int(total_token_num * self.budgets))
        all_indices = learned_scores.topk(dominant_num, dim=0).indices   # get topk indices
        all_indices = all_indices.sort().values
        hidden_states_new = hidden_states[all_indices,:]
        self.last_combined_scores = topk(learned_scores.unsqueeze(0), dominant_num).squeeze(0)
        self.last_selected_indices = all_indices
        # ----------------------------------------------------------------------------------------

        return hidden_states_new, all_indices, total_token_num

class Qwen2_5_VLForConditionalGeneration_Selector(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel_Selector._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        try:
            has_eval_time = os.environ['EVAL_TIME']
        except:
            has_eval_time = None
        if has_eval_time and os.environ['EVAL_TIME'].lower() == 'true':
            start = torch.cuda.Event(enable_timing=True)
            start.record()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, all_indices, visual_token_num = self.visual(pixel_values, grid_thw=image_grid_thw)
                origin_image_indices = torch.where(input_ids == self.config.image_token_id)[1]
                retain_image_indices = origin_image_indices[all_indices]
                origin_text_indices = torch.where(input_ids != self.config.image_token_id)[1]
                combined_indices = torch.cat((retain_image_indices, origin_text_indices))
                selected_indices, _ = torch.sort(combined_indices)

                origin_input_ids = deepcopy(input_ids)
                input_ids = input_ids[:,selected_indices]
                inputs_embeds = inputs_embeds[:,selected_indices,:]

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds, all_indices, visual_token_num = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = video_embeds.shape[0]

                total_len = input_ids.shape[-1]   
                assert input_ids.shape[0] == 1, 'selector only support single batch, assert is in qwen2vl_generation_forward_selector function'
                position_video_begin_token = (input_ids[0] == 151652).nonzero(as_tuple=True)[0]
                before_idx = position_video_begin_token[0].item() + 1   
                before_vid_idx = input_ids[:, :before_idx]
                position_video_end_token = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
                post_idx = position_video_end_token[-1].item() 
                post_vid_idx = input_ids[:, post_idx:]
                vid_tensor = torch.full(
                    (input_ids.shape[0], n_video_tokens), 
                    self.config.video_token_id, 
                    dtype=input_ids.dtype, 
                    device=input_ids.device
                )
                origin_input_ids = deepcopy(input_ids)   
                input_ids = torch.cat((before_vid_idx, vid_tensor, post_vid_idx), dim=1)  
                all_indices = all_indices + before_idx  
                combined_indices = torch.cat((torch.arange(0, before_idx, device=all_indices.device), all_indices, torch.arange(post_idx, total_len, device=all_indices.device)))  
                selected_indices, _ = torch.sort(combined_indices)
                inputs_embeds = inputs_embeds[:, selected_indices, :]

                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_mask = video_mask.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                text_image_mask = (input_ids != self.config.video_token_id)
                self.base_model.text_image_mask = text_image_mask
                for layer in self.base_model.layers:
                    layer.self_attn.text_image_mask = text_image_mask

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    origin_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                position_ids = position_ids[:,:,selected_indices]
                attention_mask = attention_mask[:,selected_indices]
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if has_eval_time and os.environ['EVAL_TIME'].lower() == 'true' and hidden_states.shape[1] != 1:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            generation_prefill_time = start.elapsed_time(end)
            print(f"Input visual token number is: {visual_token_num}")
            print(f"Generation prefill time is: {generation_prefill_time}")

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )