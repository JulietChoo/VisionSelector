from llavaonevision1_5.modeling_llavaonevision1_5 import *
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from llavaonevision1_5.modeling_llavaonevision1_5 import (
    LLaVAOneVision1_5_ModelOutputWithPast, 
    LLaVAOneVision1_5_CausalLMOutputWithPast,
    RiceTransformerPretrainedModel,
    LLaVAOneVision1_5_Model,
    LLaVAOneVision1_5_TextModel,
    LLaVAOneVision1_5_ForConditionalGeneration,
    RicePatchEmbed,
    RiceRotaryEmbedding,
    RiceBlock,
    RicePatchMerger,
)
from llavaonevision1_5.configuration_llavaonevision1_5 import Llavaonevision1_5Config, LLaVAOneVision1_5_TextConfig, RiceConfig
from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, is_torchdynamo_compiling, logging
from compression_method.selector_scorer import TransformerScorer
from torch import vmap
from torch.func import grad
from torch.autograd import Function
from copy import deepcopy

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

class RiceTransformerPretrainedModel_Selector(RiceTransformerPretrainedModel):
    config_class = RiceConfig
    _no_split_modules = ["RiceBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = RicePatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = RiceRotaryEmbedding(head_dim // 2)

        scale = config.hidden_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))
        # self.window_size = config.window_size
        self.window_size = None

        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList(
            [RiceBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = RicePatchMerger(
            dim=config.text_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size, layer_norm_eps = config.layer_norm_eps
        )
        self.gradient_checkpointing = False
        # import pdb; pdb.set_trace()
        self.importance_scorer = TransformerScorer(in_features=4096,hidden_dim=2048)


    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, is_verifying: bool=False) -> torch.Tensor:
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



class LLaVAOneVision1_5_Model_Selector(LLaVAOneVision1_5_Model):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}

    def __init__(self, config: Llavaonevision1_5Config):
        super().__init__(config)
        self.visual = RiceTransformerPretrainedModel_Selector._from_config(config.vision_config)
        self.language_model = LLaVAOneVision1_5_TextModel._from_config(config.text_config)
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


class LLaVAOneVision1_5_ForConditionalGeneration_Selector(LLaVAOneVision1_5_ForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LLaVAOneVision1_5_Model_Selector(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

