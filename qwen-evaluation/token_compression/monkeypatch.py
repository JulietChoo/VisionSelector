      
import sys
import transformers
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2, Qwen2_5_VLModel, Qwen2_5_VLDecoderLayer

import types 
import qwen25vl
import torch

from .visionzip import (
    qwen25vl_vision_flash_attention2_forward_visionzip,
    qwen25vl_vision_tower_forward_visionzip,
    qwen25vl_vision_block_forward_visionzip,
    qwen25vl_generation_forward_visionzip
)

from .fastv import (
    qwen25vl_flash_attention_forward_fastv,
    qwen25vl_model_forward_fastv,
)

from .prumerge import (
    qwen25vl_vision_flash_attention2_forward_prumerge_plus,
    qwen25vl_vision_tower_forward_prumerge_plus,
    qwen25vl_vision_block_forward_prumerge_plus,
    qwen25vl_generation_forward_prumerge_plus,
)


from .divprune import (
    qwen25vl_vision_tower_forward_divprune,
    qwen25vl_generation_forward_divprune
)

from .dart import (
    qwen25vl_flash_attention_forward_dart,
    qwen25vl_decoder_layer_forward_dart,
    get_retained_image_token,
    qwen25vl_model_forward_dart,
)

def replace_qwen25vl(args, model, method):
    if method == 'visionzip':
        print('using visionzip')
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionFlashAttention2.forward = qwen25vl_vision_flash_attention2_forward_visionzip
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen25vl_vision_tower_forward_visionzip
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock.forward = qwen25vl_vision_block_forward_visionzip
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.budgets = getattr(
            args, 'budgets', 1.0)
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.contextual_ratio = getattr(
            args, 'contextual_ratio', 0.05)
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen25vl_generation_forward_visionzip

    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLFlashAttention2):
                module.forward = types.MethodType(qwen25vl_flash_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2_5_VLModel):
                module.forward = types.MethodType(qwen25vl_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)

    elif method == 'prumerge+':
        print('using prumerge+')
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionFlashAttention2.forward = qwen25vl_vision_flash_attention2_forward_prumerge_plus
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen25vl_vision_tower_forward_prumerge_plus
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock.forward = qwen25vl_vision_block_forward_prumerge_plus
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen25vl_generation_forward_prumerge_plus

    elif method == 'divprune':
        print('using divprune')
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward = qwen25vl_vision_tower_forward_divprune
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen25vl_generation_forward_divprune
        qwen25vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'dart':
        print('using dart')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2_5_VLFlashAttention2):
                module.forward = types.MethodType(qwen25vl_flash_attention_forward_dart, module)
            if isinstance(module, Qwen2_5_VLDecoderLayer):
                module.forward = types.MethodType(qwen25vl_decoder_layer_forward_dart, module)
            if isinstance(module, Qwen2_5_VLModel):
                module.forward = types.MethodType(qwen25vl_model_forward_dart, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)