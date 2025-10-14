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
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from copy import deepcopy
from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available
if is_flash_attn_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from flash_attn import flash_attn_varlen_func


def llavaov15_flash_attention_forward_fastv(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    if self.layer_idx == self.target_layer_idx - 1:   # HACK add here
        q_len = hidden_states.shape[1]
        attn_weights_here = torch.matmul(query_states.float(), key_states.transpose(2, 3).float()) / math.sqrt(self.head_dim)
        attn_mask = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1
        )
        attn_weights_here = attn_weights_here.masked_fill(attn_mask, float('-inf'))
        attn_weights_here = nn.functional.softmax(attn_weights_here, dim=-1, dtype=torch.float32)
        self.last_attention = attn_weights_here
        del attn_weights_here, attn_mask

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        input_shape[1],
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value    


def llavaov15_language_model_forward_fastv(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch_size, seq_length = inputs_embeds.shape[:2]    

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    # elif position_ids.dim() == 2: # 这是为了3drope准备的
    #     position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            if decoder_layer.self_attn.layer_idx == self.target_layer_idx and seq_length > 1:
                ratio = self.budgets
                assert ratio is not None, "budgets is None, please pass it"
                assert batch_size == 1, "fastv only support single batch but the batch size here > 1"
                last_attention = self.layers[self.target_layer_idx - 1].self_attn.last_attention
                assert last_attention is not None, 'last attension is None, but it should be not None'

                text_image_mask = self.text_image_mask[0]
                image_start = int((text_image_mask == False).nonzero(as_tuple=True)[0][0])
                image_end = int((text_image_mask == False).nonzero(as_tuple=True)[0][-1])   
                image_length = image_end - image_start + 1
                assert image_start is not None and image_end is not None, "no image token found, fastv here now is must to have an image"   # at target layer and in prefill stage
                
                device = hidden_states.device
                if self.origin:
                    image_attention_score = last_attention.mean(dim=1)[0][-1][image_start:image_end + 1]    # FIXME 
                else:
                    image_attention_score = last_attention.mean(dim=1)[0][:, image_start:image_end + 1].mean(dim=0)    # FIXME 
                top_attention_rank_index = image_attention_score.topk(max(1, int(image_length * ratio))).indices + image_start     
                keep_indexs = torch.cat((torch.arange(image_start,device=device), top_attention_rank_index, torch.arange(image_length+image_start,seq_length,device=device)))
                keep_indexs = keep_indexs.sort().values
                hidden_states = hidden_states[:,keep_indexs,:]
                if causal_mask is not None:
                    causal_mask = causal_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                position_ids = position_ids[:, keep_indexs]      
                position_embeddings = self.rotary_emb(hidden_states, position_ids)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )