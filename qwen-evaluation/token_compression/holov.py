import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from qwen25vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
import os


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
logger = logging.get_logger(__name__)
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, apply_rotary_pos_emb_flashatt
from copy import deepcopy

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # 确保 attn_bias 的维度与 attn_mask 匹配
            if attn_mask.dim() > attn_bias.dim():
                # 如果 attn_mask 维度更高，需要扩展 attn_bias
                attn_bias = attn_bias.unsqueeze(0).expand_as(attn_mask)
            elif attn_mask.dim() < attn_bias.dim():
                # 如果 attn_mask 维度更低，需要压缩 attn_mask
                attn_mask = attn_mask.unsqueeze(0)
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_logits = attn_weight
    return attn_weight @ value,attn_logits





def HoloV(image_tokens, attention, num_patches, new_image_token_num, esp=1e-6):
    attention = attention.unsqueeze(0)
    B, N, D = image_tokens.shape
    device = image_tokens.device
    alpha = 1
    beta = 0.09
    power = 1
    pruned_image_tokens_list = []
    final_positions_list = []  

    for b in range(B):
        image_token = image_tokens[b]  # [N, D]
        image_attention = attention[b]  # [N]

        # Calculate dynamic patch size - handle uneven divisions
        patch_size = N // num_patches
        remainder = N % num_patches

        # Create patches with potentially uneven sizes
        image_tokens_patches = []
        attention_patches = []
        start_idx = 0
        patch_start_indices = []  

        for p in range(num_patches):
            # Last few patches get an extra token if there's a remainder
            current_patch_size = patch_size + (1 if p < remainder else 0)
            end_idx = start_idx + current_patch_size

            if current_patch_size > 0:  # Skip empty patches
                image_tokens_patches.append(image_token[start_idx:end_idx])
                attention_patches.append(image_attention[start_idx:end_idx])
                patch_start_indices.append(start_idx)  # 记录patch起始位置

            start_idx = end_idx

        # Process each patch separately
        patch_scores = []
        all_patches = []
        all_patch_indices = []  

        for p in range(len(image_tokens_patches)):
            patch_tokens = image_tokens_patches[p]  # [current_patch_size, D]
            patch_attn = attention_patches[p]  # [current_patch_size]
            patch_start = patch_start_indices[p]  
            current_patch_size = len(patch_tokens)

            
            patch_indices = torch.arange(patch_start, patch_start + current_patch_size, device=device)

            if current_patch_size <= 1:
                # If patch has only one token or is empty, handle specially
                patch_scores.append(patch_attn.mean() if len(patch_attn) > 0 else torch.tensor(0.0, device=device))
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)
                continue

            with torch.no_grad():
                # Normalize patch tokens
                F_normalized = patch_tokens / (patch_tokens.norm(dim=1, keepdim=True) + esp)

                # Compute similarity matrix
                S = torch.mm(F_normalized, F_normalized.transpose(0, 1))

                # Create eye mask of appropriate size
                eye_mask = 1 - torch.eye(current_patch_size, device=device)
                S_masked = S * eye_mask

                # Compute mean and variance
                valid_entries = current_patch_size - 1
                mean_sim = S_masked.sum(dim=1) / valid_entries
                var_sim = ((S_masked - mean_sim.unsqueeze(1))**2).sum(dim=1) / valid_entries

                # Scale attention
                patch_attn_scaled = patch_attn * 1e3

                # Scale variance
                var_scaling = (torch.mean(torch.abs(patch_attn_scaled)) / 
                              (torch.mean(torch.abs(var_sim)) + esp))
                var_sim_scaled = var_sim * var_scaling

                # Calculate token scores
                token_scores = alpha * patch_attn_scaled + beta * var_sim_scaled

                # Compute patch score
                patch_score = token_scores.mean()
                patch_scores.append(patch_score)
                all_patches.append(patch_tokens)
                all_patch_indices.append(patch_indices)

        # Convert to tensor
        patch_scores = torch.stack(patch_scores) if patch_scores else torch.zeros(0, device=device)

        # Allocate new tokens based on scores
        if len(patch_scores) > 0:
            weights = (patch_scores ** power) / ((patch_scores ** power).sum() + esp)
            allocated = (weights * new_image_token_num).floor().long()

            # Distribute remaining tokens
            remaining = new_image_token_num - allocated.sum()
            if remaining > 0 and len(weights) > 0:
                _, indices = torch.topk(weights, k=min(remaining.item(), len(weights)))
                for idx in indices[:remaining]:
                    allocated[idx] += 1

            # Handle token overflow
            new_patches = []
            final_positions = []  

            for i, (patch, alloc, patch_indices) in enumerate(zip(all_patches, allocated, all_patch_indices)):
                patch_size = len(patch)
                if alloc <= 0:
                    continue
                elif alloc >= patch_size:
                    # Keep all tokens in this patch
                    new_patches.append(patch)
                    final_positions.append(patch_indices)
                else:
                    # Sample tokens based on attention scores
                    patch_attn = attention_patches[i]
                    _, top_indices = torch.topk(patch_attn, k=min(alloc.item(), patch_size))
                    new_patches.append(patch[top_indices])
                    final_positions.append(patch_indices[top_indices])

            # Combine all selected tokens
            if new_patches:
                new_image_tokens = torch.cat(new_patches, dim=0)
                final_positions = torch.cat(final_positions, dim=0) 
            else:
                new_image_tokens = torch.zeros((0, D), device=device)
                final_positions = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # No patches to process
            new_image_tokens = torch.zeros((0, D), device=device)
            final_positions = torch.zeros(0, dtype=torch.long, device=device)

        # Pad or truncate to match expected new_image_token_num
        actual_tokens = new_image_tokens.size(0)
        if actual_tokens < new_image_token_num:
            # Pad with zeros if we don't have enough tokens
            padding = torch.zeros((new_image_token_num - actual_tokens, D), device=device)
            new_image_tokens = torch.cat([new_image_tokens, padding], dim=0)
            
            
            padding_positions = torch.full((new_image_token_num - actual_tokens,), -1, dtype=torch.long, device=device)
            final_positions = torch.cat([final_positions, padding_positions], dim=0)
        elif actual_tokens > new_image_token_num:
            # Truncate if we have too many tokens
            new_image_tokens = new_image_tokens[:new_image_token_num]
            final_positions = final_positions[:new_image_token_num]

        pruned_image_tokens_list.append(new_image_tokens)
        final_positions_list.append(final_positions)

    
    return torch.stack(pruned_image_tokens_list, dim=0), torch.stack(final_positions_list, dim=0).squeeze(0)







def adjust_ids(input_ids, position_ids, image_token_start, image_token_len, final_position):
    
    if input_ids is not None:
        pre_ids = input_ids[:, :image_token_start]
        post_ids = input_ids[:, image_token_start + image_token_len:]
        kept_ids = input_ids[:, image_token_start:image_token_start + image_token_len]
        kept_ids = kept_ids[:, final_position]  
        adjusted_input_ids = torch.cat([pre_ids, kept_ids, post_ids], dim=1)
    else:
        adjusted_input_ids = None
    
    if position_ids is not None:
        pre_pos = position_ids[:, :, :image_token_start]
        post_pos = position_ids[:, :, image_token_start + image_token_len:]
        kept_pos = position_ids[:, :, image_token_start:image_token_start + image_token_len]
        kept_pos = kept_pos[:, :, final_position]  
        adjusted_position_ids = torch.cat([pre_pos, kept_pos, post_pos], dim=2)
    else:
        adjusted_position_ids = None
        
    return adjusted_input_ids, adjusted_position_ids


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def qwen25vl_vision_flash_attention2_forward_holov(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        if output_attentions:
            k_here = k.transpose(0, 1)   # [num_heads, seq_len, head_dim]
            q_here = q.transpose(0, 1)
            attn_weights_here = torch.matmul(q_here, k_here.transpose(1, 2)) / math.sqrt(q.shape[-1])   # [num_heads, seq_len, seq_len]
            attention_mask_here = torch.full( 
                [1, seq_length, seq_length], True, dtype=torch.bool, device=q.device
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask_here[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = False 
            attn_weights_here = attn_weights_here.masked_fill(attention_mask_here, float('-inf'))
            del k_here, q_here,attention_mask_here
            attn_weights = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        else:
            attn_weights = None

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights
        

def qwen25vl_vision_sdpa_attention_forward_holov(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
    
        # attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        if output_attentions:
            attn_output, attn_weights = scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
            )
        else:
            # 使用SDPA计算注意力输出
            attn_output = F.scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
            )
            attn_weights = None
            
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights


def qwen25vl_vision_block_forward_holov(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        attn_output, attn_weights = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, attn_weights


def qwen25vl_vision_tower_forward_holov(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
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

        last_attention_weights = None

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            output_attentions = True
            output_last_layer_attention = output_attentions and (layer_num == len(self.blocks) - 1)
            
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states, attn_weights = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings, output_attentions=output_last_layer_attention)

            if output_last_layer_attention:
                last_attention_weights = attn_weights
        
        attention_scores = None
        if last_attention_weights is not None:
            if last_attention_weights.dim() == 3:  # [num_heads, seq_len, seq_len]
                attention_weights_mean = last_attention_weights.mean(dim=0)  # [seq_len, seq_len]
                attention_scores = attention_weights_mean.mean(dim=0)  # [seq_len]
            elif last_attention_weights.dim() == 4:  # [1, num_heads, seq_len, seq_len]
                attention_weights_mean = last_attention_weights.mean(dim=1)  # [num_heads, seq_len, seq_len]
                attention_scores = attention_weights_mean.mean(dim=1)  
            else:
                attention_scores = torch.ones(seq_len, device=hidden_states.device)
        
        pre_merge_seq_len = hidden_states.shape[0]
        
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        if attention_scores is not None:
            post_merge_seq_len = hidden_states.shape[0]
            merge_ratio = pre_merge_seq_len // post_merge_seq_len
            
            if merge_ratio > 1:
                attention_scores = attention_scores[:post_merge_seq_len * merge_ratio]
                attention_scores = attention_scores.view(post_merge_seq_len, merge_ratio)
                attention_scores = attention_scores.mean(dim=1)

        return hidden_states, attention_scores


def qwen25vl_generation_forward_holov(
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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
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
                image_embeds, image_attn = self.visual(pixel_values, grid_thw=image_grid_thw)
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                text_image_mask = (input_ids != 151655)   # HACK  change here to get image_mask(image position is false, else is true)
                self.base_model.text_image_mask = text_image_mask
                for layer in self.base_model.layers:
                        layer.self_attn.text_image_mask = text_image_mask

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

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
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
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
        # print('inputs_embeds.shape:',inputs_embeds.shape)

        # ##############################change###########################################
        # if pixel_values is not None:
        #     image_token_end = image_token_start+image_token_len
        #     pre_prompt = inputs_embeds[:,:image_token_start,:]
        #     image_tokens = inputs_embeds[:,image_token_start:image_token_end,:]
        #     others = inputs_embeds[:,image_token_end:,:]
            
            
        #     prune_rate = self.budgets
        #     keep_num = int(image_token_len * prune_rate)
        #     num_patches = int(((1024/576)*image_token_len)/keep_num)
        #     num_patches = max(1,int(((1024/576)*image_token_len)/keep_num))
        #     new_image_tokens, final_position = HoloV(image_tokens, image_attn, num_patches, keep_num)

        #     new_input_embeds = torch.cat([pre_prompt,new_image_tokens,others],dim=1)
        #     inputs_embeds = new_input_embeds.to(pre_prompt.dtype)
        #     input_ids, position_ids = adjust_ids(input_ids, position_ids, image_token_start, image_token_len, final_position)
        # ##############################change###########################################

        ##############################change - 支持多图###########################################
        if pixel_values is not None:
            # 找到所有图像token的起始位置
            image_token_starts = torch.where(input_ids[0] == self.config.vision_start_token_id)[0]
            
            if len(image_token_starts) > 0:
                # 计算每个图像的token数量
                total_image_tokens = image_embeds.shape[0]
                num_images = len(image_token_starts)
                
                # 假设所有图像的token长度相同（根据image_grid_thw可以更精确计算）
                if image_grid_thw is not None and len(image_grid_thw) == num_images:
                    # 根据grid_thw计算每个图像的实际token数量
                    tokens_per_image_list = []
                    for i in range(num_images):
                        t, h, w = image_grid_thw[i]
                        # 计算该图像的token数量（这里需要根据实际的patch embedding逻辑）
                        # 简化假设：按比例分配
                        tokens_per_image_list.append(h * w * t)
                    
                    # 归一化使总和等于total_image_tokens
                    total_calculated = sum(tokens_per_image_list)
                    tokens_per_image_list = [int(t * total_image_tokens / total_calculated) for t in tokens_per_image_list]
                    
                    # 处理舍入误差
                    diff = total_image_tokens - sum(tokens_per_image_list)
                    if diff > 0:
                        tokens_per_image_list[-1] += diff
                else:
                    # 如果没有grid_thw信息，平均分配
                    tokens_per_image = total_image_tokens // num_images
                    tokens_per_image_list = [tokens_per_image] * num_images
                    # 处理余数
                    remainder = total_image_tokens % num_images
                    for i in range(remainder):
                        tokens_per_image_list[i] += 1
                
                # 累积处理每个图像
                current_inputs_embeds = inputs_embeds
                current_input_ids = input_ids
                current_position_ids = position_ids
                offset = 0  # 跟踪由于token数量变化导致的位置偏移
                image_attn_offset = 0  # 跟踪image_attn中的偏移
                
                for img_idx in range(num_images):
                    # 获取当前图像在序列中的起始位置（考虑之前图像处理导致的偏移）
                    image_token_start = image_token_starts[img_idx].item() + offset
                    image_token_len = tokens_per_image_list[img_idx]
                    image_token_end = image_token_start + image_token_len
                    
                    # 提取当前图像的tokens和对应的attention scores
                    pre_prompt = current_inputs_embeds[:, :image_token_start, :]
                    image_tokens = current_inputs_embeds[:, image_token_start:image_token_end, :]
                    others = current_inputs_embeds[:, image_token_end:, :]
                    
                    # 提取对应的注意力分数
                    image_attn_curr = image_attn[image_attn_offset:image_attn_offset + image_token_len]
                    image_attn_offset += image_token_len
                    
                    # 应用HoloV压缩
                    prune_rate = self.budgets
                    keep_num = int(image_token_len * prune_rate)
                    keep_num = max(1, keep_num)  # 至少保留1个token
                    
                    num_patches = max(1, int(((1024/576) * image_token_len) / keep_num))
                    
                    new_image_tokens, final_position = HoloV(
                        image_tokens, 
                        image_attn_curr, 
                        num_patches, 
                        keep_num
                    )
                    
                    # 重新组合embeddings
                    current_inputs_embeds = torch.cat([pre_prompt, new_image_tokens, others], dim=1)
                    
                    # 调整input_ids和position_ids
                    current_input_ids, current_position_ids = adjust_ids(
                        current_input_ids, 
                        current_position_ids, 
                        image_token_start, 
                        image_token_len, 
                        final_position
                    )
                    
                    # 更新偏移量（当前图像压缩导致的序列长度变化）
                    offset += (keep_num - image_token_len)
                
                # 更新最终的embeddings和ids
                inputs_embeds = current_inputs_embeds.to(inputs_embeds.dtype)
                input_ids = current_input_ids
                position_ids = current_position_ids
        ##############################change###########################################
            
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
            if pixel_values is not None:
                visual_token_num = n_image_tokens
            elif pixel_values_videos is not None:
                visual_token_num = n_video_tokens
            else:
                visual_token_num = 0
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