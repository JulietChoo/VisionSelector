import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llavaonevision1_5.modeling_llavaonevision1_5 import *
from typing import Any, Dict, List, Optional, Tuple, Union
from llavaonevision1_5.modeling_llavaonevision1_5 import (
    LLaVAOneVision1_5_ModelOutputWithPast, 
    LLaVAOneVision1_5_CausalLMOutputWithPast,
    apply_rotary_pos_emb_vision,
)
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from copy import deepcopy
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from flash_attn import flash_attn_varlen_func

def llavaov15_vision_flash_attention_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 22:
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
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    if self.layer_idx == 22:
        k_here = k.transpose(0, 1)   # [num_heads, seq_len, head_dim]
        self.metric = k_here
        q_here = q.transpose(0, 1)
        attn_weights_here = torch.matmul(q_here, k_here.transpose(1, 2)) / math.sqrt(q_here.shape[-1])   # [num_heads, seq_len, seq_len]
        attn_weights_here = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        self.attn_weights = attn_weights_here
        del k_here, q_here, attn_weights_here

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        seq_length, -1
    )
    attn_output = self.proj(attn_output)
    return attn_output


def llavaov15_vision_block_forward_visionzip(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if self.layer_idx == 22:
        self.hidden_states = None
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    if self.layer_idx == 22:
        self.hidden_states = hidden_states
    return hidden_states


def llavaov15_vision_tower_forward_visionzip(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, is_verifying: bool=False) -> torch.Tensor:
    r"""
    grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
        The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    img_feats = hidden_states.shape[0]
    
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    cu = cu_seqlens.to(torch.long)
    num_segments = cu.numel() - 1
    cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)

    total_patches = cu[-1].item()
    new_total = total_patches + num_segments
    D = hidden_states.size(-1)
    new_hidden = hidden_states.new_empty((new_total, D))
    new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

    write_ptr = 0
    new_cu = [0]
    for i in range(1, num_segments + 1):
        seg_start = cu[i-1].item()
        seg_end = cu[i].item()
        seg_len = seg_end - seg_start
        new_hidden[write_ptr] = cls_token
        new_rotary_pos_emb[write_ptr] = self.class_pos_emb
        new_hidden[write_ptr + 1: write_ptr + 1 + seg_len] = hidden_states[seg_start:seg_end]
        new_rotary_pos_emb[write_ptr + 1: write_ptr + 1 + seg_len] = rotary_pos_emb[seg_start:seg_end]
        write_ptr += 1 + seg_len
        new_cu.append(write_ptr)

    hidden_states = new_hidden
    cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32) 
    rotary_pos_emb = new_rotary_pos_emb

    hidden_states = self.pre_layernorm(hidden_states)

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    for blk in self.blocks:
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    
    new_hidden = hidden_states.new_empty((img_feats, D))

    for i in range(1, num_segments + 1):
        seg_start = cu[i-1].item()
        seg_end = cu[i].item()
        new_hidden[seg_start:seg_end] = hidden_states[seg_start+1:seg_end+1]
    hidden_states = new_hidden
    if is_verifying:
        return hidden_states

    hidden_states = self.merger(hidden_states)
    # return hidden_states
    #####add###########
    image_feature_length = hidden_states.shape[0]
    attn_weights = self.blocks[-2].attn.attn_weights   # shape:torch.Size([16, 2320, 2320])
    metric = self.blocks[-2].attn.metric  # shape: torch.Size([16, 2320, 80])
    num_heads, q_len, k_len = attn_weights.shape
    assert q_len == k_len, "q_len and k_len should be the same, the error is in Qwen2VisionTransformerPretrainedModel's forward function"
    # attn_weights = attn_weights.mean(dim=0).mean(dim=0)  # shape: torch.Size([2320]) 
    attn_weights = attn_weights.mean(dim=0)
    new_attn_weights = attn_weights.new_empty((img_feats))
    new_metric = metric.new_empty((metric.shape[0], img_feats, metric.shape[2]))
    for i in range(1, num_segments + 1):
        seg_start = cu[i-1].item()
        seg_end = cu[i].item()
        new_attn_weights[seg_start:seg_end] = attn_weights[i-1,seg_start+1:seg_end+1]
        new_metric[:,seg_start:seg_end,:] = metric[:,seg_start+1:seg_end+1,:]
    attn_weights = new_attn_weights
    metric = new_metric
    try:
        attention_sum = attn_weights.view(-1, 4).mean(dim=1) # shape: torch.Size([580]
        metric = metric.view(num_heads, metric.shape[1] // 4, 4, -1)  # shape: torch.Size([16, 580, 4, 80])
    except:
        import pdb; pdb.set_trace()
    self.blocks[-2].attn.metric = None
    metric = metric.mean(dim=2).mean(dim=0)   # shape:torch.Size([580, 80])
    total_token_num = metric.shape[0]
    ########add#########
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
    ###################
    return hidden_states_save, all_keep_indices, image_feature_length


def llavaov15_vlmodel_forward_visionzip(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
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
) -> Union[Tuple, LLaVAOneVision1_5_ModelOutputWithPast]:
    r"""
    pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
        The tensors corresponding to the input videos. Pixel values can be obtained using
        [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
        [`Qwen2VLImageProcessor`] for processing videos.
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    visual_token_num = 0
    selected_indices = []
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds, all_indices, total_token_num = self.get_image_features(pixel_values, image_grid_thw)
            # n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            # n_image_features = image_embeds.shape[0]
            # if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            #     raise ValueError(
            #         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            #     )
            # image_mask = (
            #     (input_ids == self.config.image_token_id)
            #     .unsqueeze(-1)
            #     .expand_as(inputs_embeds)
            #     .to(inputs_embeds.device)
            # )
            # image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            # inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
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
            visual_token_num = total_token_num

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            visual_token_num = n_video_tokens

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    # add
    if inputs_embeds.shape[1] != 1:
        cache_position = cache_position[selected_indices]
        position_ids = position_ids[:,selected_indices]
        attention_mask = attention_mask[:,selected_indices]

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
    )

    output = LLaVAOneVision1_5_ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple(), visual_token_num