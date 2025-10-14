import torch
import torch.nn.functional as F
from llavaonevision1_5.modeling_llavaonevision1_5 import *
from typing import Any, Dict, List, Optional, Tuple, Union
from llavaonevision1_5.modeling_llavaonevision1_5 import (
    LLaVAOneVision1_5_ModelOutputWithPast, 
    LLaVAOneVision1_5_CausalLMOutputWithPast
)
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from copy import deepcopy

def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity


def DivPrune(visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1):            
    threshold_terms = int(round(threshold_ratio*image_feature_length))
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    returned_scores = torch.topk(cosine_matrix, 2, dim=0, largest=False).values[1, :] if image_feature_length > 1 else torch.zeros(image_feature_length, device=visual_feature_vectors.device)
    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i==0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

        if i==0:
            scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
        else:
            scores = torch.min(m2, dim=0).values #for distance 

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s, cosine_matrix, returned_scores

def llavaov15_vision_tower_forward_divprune(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, is_verifying: bool=False) -> torch.Tensor:
    "RiceTransformerPretrainedModel.forward"
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
    # ---------------------------add---------------------------------------------
    cosine_matrix = None
    image_feature_length = hidden_states.shape[0]
    selected_indices, cosine_matrix, divprune_scores = DivPrune(hidden_states, image_feature_length, cosine_matrix, threshold_ratio=self.budgets)
    selected_indices = selected_indices.sort().values
    hidden_states_save = hidden_states[selected_indices,:]
    self.last_divprune_scores = divprune_scores
    self.last_selected_indices = selected_indices
    # -------------------------------------------------------------
    return hidden_states_save, selected_indices, image_feature_length


def llavaov15_vlmodel_forward_divprune(
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