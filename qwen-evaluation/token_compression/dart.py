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
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, apply_rotary_pos_emb_flashatt, apply_multimodal_rotary_pos_emb, repeat_kv
from copy import deepcopy

def qwen25vl_flash_attention_forward_dart(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

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
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value, query_states.permute(0, 2, 1, 3), key_states.permute(0, 2, 1, 3), value_states.permute(0, 2, 1, 3)


def qwen25vl_decoder_layer_forward_dart(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, query_states, key_states, value_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        outputs += (query_states, key_states, value_states)

        return outputs


def get_retained_image_token(  
    last_layer_state: torch.Tensor,  
    any_states: torch.Tensor,  
    text_image_mask: torch.Tensor,  
    visual_token_indices: torch.Tensor,  
    retain_ratio: float  
) -> torch.Tensor:  
    """  
    Calculates which image tokens to retain based on L1 norm and cosine similarity.  
    This corrected version ensures query tokens are only selected from *after* the last visual token.  

    Args:  
        last_layer_state: The hidden states from the last layer.  
        any_states: The key states from an attention layer.  
        text_image_mask: A boolean mask where True indicates a text token and False indicates a visual token.  
        visual_token_indices: The indices of the visual tokens in the sequence.  
        retain_ratio: The ratio of image tokens to retain.  

    Returns:  
        A tensor containing the indices of the image tokens to be retained.  
    """  
    pivot_image_token = 4  
    pivot_text_token = 4  

    image_token_length = visual_token_indices.shape[0]  
    if image_token_length == 0:  
        return torch.tensor([], device=last_layer_state.device, dtype=torch.long)  
        
    TOKEN_TOPK = max(1, int(image_token_length * retain_ratio / (pivot_image_token + pivot_text_token)))  

    device = last_layer_state.device  

    any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.shape[0], any_states.shape[2], -1)  
    seq_length = any_states.shape[1]  

    # --- Corrected Logic ---  
    # Find the index of the last visual token to define the start of the query section.  
    last_visual_token_index = visual_token_indices.max()  
    query_token_start_index = last_visual_token_index + 1  

    # Image tokens are selected using the mask for safety with non-contiguous tokens.  
    k_states_image_token = any_states[0][~text_image_mask, :]  
    
    # Query tokens are ONLY those appearing *after* the final image token.  
    k_states_query_token = any_states[0][query_token_start_index:, :]  
    # -------------------------  

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)  
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)  

    # --- Corrected Index Mapping ---  
    # Map relative image topk indices back to the original absolute indices.  
    top_image_indices_relative = k_states_image_token_L1_norm.topk(min(pivot_image_token, image_token_length)).indices  
    image_indices = visual_token_indices[top_image_indices_relative].tolist()  

    # Create an index tensor for the query tokens to map them back correctly.  
    query_indices = []  
    if k_states_query_token.shape[0] > 0:  
        query_token_indices = torch.arange(query_token_start_index, seq_length, device=device)  
        top_query_indices_relative = k_states_query_token_L1_norm.topk(min(pivot_text_token, len(query_token_indices))).indices  
        query_indices = query_token_indices[top_query_indices_relative].tolist()  
    # -------------------------------  
    
    indices_set = set(image_indices + query_indices)  

    valid_indices = set(visual_token_indices.tolist()) - set(image_indices)  
    valid_indices_list = list(valid_indices)  

    for item in list(indices_set):  
        if not valid_indices_list:  
            break  
        
        valid_vectors = last_layer_state[0][valid_indices_list, :]  
        cos_sim = -torch.nn.functional.cosine_similarity(last_layer_state[0][item, :], valid_vectors, dim=-1)  
        
        current_topk = min(TOKEN_TOPK, len(valid_indices_list))  
        if current_topk == 0:  
            continue  

        top_k_indices = cos_sim.topk(current_topk).indices  
        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]  
        
        indices_set.update(top_k_real_indices)  
        valid_indices.difference_update(top_k_real_indices)  
        valid_indices_list = list(valid_indices)  

    # The function should only return retained *image* tokens.  
    indices_set.difference_update(query_indices)  

    retained_image_tokens_index = torch.tensor(list(indices_set), device=device, dtype=torch.long)  

    return retained_image_tokens_index   


def qwen25vl_model_forward_dart(
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
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

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
            if decoder_layer.self_attn.layer_idx == self.target_layer_idx and seq_length > 1 and hasattr(self, 'text_image_mask') and self.text_image_mask is not None:  
                device = hidden_states.device  
                last_layer_state = self.norm(hidden_states)  
                k_states = layer_outputs[-2]  
                text_image_mask = self.text_image_mask[0] # Shape: [current_seq_length]  

                visual_token_indices = torch.where(text_image_mask == False)[0]  
                text_token_indices = torch.where(text_image_mask == True)[0]  

                if visual_token_indices.shape[0] > 0:  
                    retained_image_tokens_index = get_retained_image_token(  
                        last_layer_state, k_states, text_image_mask, visual_token_indices, self.budgets  
                    ).to(device)  
                    keep_indices = torch.cat((text_token_indices, retained_image_tokens_index)).sort().values  

                    # Prune all tensors that are passed between layers.  
                    hidden_states = hidden_states[:, keep_indices, :]  
                    if causal_mask is not None:  
                        # Prune both dimensions of the attention mask.  
                        causal_mask = causal_mask[:, :, keep_indices, :][:, :, :, keep_indices]  
                    
                    position_ids = position_ids[:, :, keep_indices]  
                    
                    # Recompute position embeddings for the new pruned sequence.  
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)  

                    # Update sequence length and other relevant tensors for the next iteration.  
                    self.text_image_mask = self.text_image_mask[:, keep_indices]  
                    seq_length = hidden_states.shape[1]  
                    if cache_position is not None:  
                        cache_position = cache_position[keep_indices]

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
