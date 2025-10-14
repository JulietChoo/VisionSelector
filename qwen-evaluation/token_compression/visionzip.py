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

def qwen25vl_vision_flash_attention2_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 31:
        self.metric = None
        self.attention_weights = None

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
    if self.layer_idx == 31:  
        k_here = k.transpose(0, 1)   # [num_heads, seq_len, head_dim]
        self.metric = k_here
        q_here = q.transpose(0, 1)

        attention_mask_here = torch.full(  
            [1, seq_length, seq_length], True, dtype=torch.bool, device=q.device
        )
        
        for i in range(1, len(cu_seqlens)):
            attention_mask_here[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = False  
        # print(attention_mask_here.sum())
        attn_weights_here = torch.matmul(q_here, k_here.transpose(1, 2)) / math.sqrt(q.shape[-1])   # [num_heads, seq_len, seq_len]
        attn_weights_here = attn_weights_here.masked_fill(attention_mask_here, float('-inf'))
 
        attn_weights_here = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        self.attn_weights = attn_weights_here
        del k_here, q_here, attention_mask_here, attn_weights_here

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output

def qwen25vl_vision_tower_forward_visionzip(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
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
    #####add###########
    attn_weights = self.blocks[-1].attn.attn_weights   # shape:torch.Size([16, 2320, 2320])
    num_heads, q_len, k_len = attn_weights.shape
    assert q_len == k_len, "q_len and k_len should be the same, the error is in Qwen2VisionTransformerPretrainedModel's forward function"
    attn_weights = attn_weights.mean(dim=0).mean(dim=0)  # shape: torch.Size([2320]) 
    attention_sum = attn_weights.view(-1, 4).mean(dim=1) # shape: torch.Size([580]
    ###################
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    ########add################
    metric = self.blocks[-1].attn.metric  # shape: torch.Size([16, 2320, 80])
    self.blocks[-1].attn.metric = None
    metric = metric.view(num_heads, metric.shape[1] // 4, 4, -1)  # shape: torch.Size([16, 580, 4, 80])

    metric = metric.mean(dim=2).mean(dim=0)   # shape:torch.Size([580, 80])
    total_token_num = metric.shape[0]

    #####新增#######
    attention_sum = attention_sum[reverse_indices]
    metric = metric[reverse_indices,:]

    if self.contextual_ratio == 0:
        # print('self.contextual_ratio is 0')
        dominant_num = max(1, int(total_token_num * self.budgets))
        all_indices = attention_sum.topk(dominant_num, dim=0).indices   # get topk indices
        all_indices = all_indices.sort().values
        all_keep_indices = all_indices
        hidden_states_save = hidden_states[all_indices,:]
        return hidden_states_save, all_keep_indices
    else:
        # print('self.contextual_ratio is not 0')
        dominant_num = max(1, int(total_token_num * (self.budgets-self.contextual_ratio)))
        contextual_num = max(1, int(total_token_num * self.contextual_ratio))
        all_indices = attention_sum.topk(dominant_num, dim=0).indices   # get topk indices
        all_indices = all_indices.sort().values

        mask = torch.ones_like(hidden_states[:, 0], dtype=torch.bool, device=metric.device).scatter_(0, all_indices, False)  # true is delete

        filtered_indices = torch.where(mask)[0]  
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(dominant_num, hidden_states.shape[1])

        metric_filtered = metric[mask].view(hidden_states.shape[0] - dominant_num, metric.shape[1]) 
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0] - dominant_num, hidden_states.shape[1])  
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        ## Start merge
        step = max(1, metric_normalized.shape[0] // contextual_num)

        target_indices = torch.arange(0, metric_normalized.shape[0], step, device=metric_normalized.device)[:contextual_num] 
        contextual_indices = filtered_indices[target_indices]  
        target_tokens = metric_normalized[target_indices, :]  


        tokens_to_merge = metric_normalized[~torch.isin(torch.arange(metric_normalized.shape[0], device=metric_normalized.device), target_indices), :]  
        similarity = torch.matmul(tokens_to_merge.float(), target_tokens.transpose(0, 1).float())  
        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(1, similarity.argmax(dim=1).unsqueeze(-1), 1)

        counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)      

        hidden_to_merge = hidden_states_filtered[~torch.isin(torch.arange(hidden_states_filtered.shape[0], device=hidden_states_filtered.device), target_indices), :]
        aggregated_hidden = (torch.matmul(assign_one_hot.transpose(0, 1).float(), hidden_to_merge.float()) / counts).to(torch.bfloat16) 
        target_hidden = hidden_states_filtered[target_indices, :] 
        
        contextual_tokens = target_hidden + aggregated_hidden


    
        all_keep_indices = torch.cat([all_indices, contextual_indices])
        all_keep_indices = all_keep_indices.sort().values 
    

        hidden_states_save = torch.zeros(
            (len(all_keep_indices), hidden_states.shape[1]), 
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        dominant_mask = torch.zeros(len(all_keep_indices), dtype=torch.bool, device=hidden_states.device)
        dominant_mask = torch.isin(all_keep_indices, all_indices)
        dominant_positions = torch.where(dominant_mask)[0]
        hidden_states_save[dominant_positions] = dominant_tokens
        contextual_positions = torch.where(~dominant_mask)[0]
        hidden_states_save[contextual_positions] = contextual_tokens
        self.last_attention_sum = attention_sum
        self.last_selected_indices = all_keep_indices

        return hidden_states_save, all_keep_indices


def qwen25vl_vision_block_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 31:
        self.hidden_states = None

    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    if self.layer_idx == 31:
        self.hidden_states = hidden_states
    return hidden_states


def qwen25vl_generation_forward_visionzip(
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

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, all_indices = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = image_embeds.shape[0] 

            total_len = input_ids.shape[-1]   
            assert input_ids.shape[0] == 1, 'visionzip only support single batch, assert is in qwen2vl_generation_forward_visionzip function'
            position_image_begin_token = (input_ids[0] == 151652).nonzero(as_tuple=True)[0]
            before_idx = position_image_begin_token[0].item() + 1   
            before_img = input_ids[:, :before_idx]
            position_image_end_token = (input_ids[0] == 151653).nonzero(as_tuple=True)[0]
            post_idx = position_image_end_token[-1].item() 
            post_img = input_ids[:, post_idx:]
            img_tensor = torch.full(
                (input_ids.shape[0], n_image_tokens), 
                self.config.image_token_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )
            origin_input_ids = deepcopy(input_ids)   
            input_ids = torch.cat((before_img, img_tensor, post_img), dim=1)  
            all_indices = all_indices + before_idx  
            all_indices = torch.cat((torch.arange(0, before_idx, device=all_indices.device), all_indices, torch.arange(post_idx, total_len, device=all_indices.device)))  
            inputs_embeds = inputs_embeds[:, all_indices, :]

            
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )

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
                origin_input_ids,  ##add
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
            ######### post handle #########
            position_ids = position_ids[:, :, all_indices]    # because we don't prune text token, so we don't need to change rope_deltas
            attention_mask = attention_mask[:, all_indices]
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