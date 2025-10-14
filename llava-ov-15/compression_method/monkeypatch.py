import types
from llavaonevision1_5.modeling_llavaonevision1_5 import (
    RiceTransformerPretrainedModel,
    LLaVAOneVision1_5_Model,
    RiceFlashAttention2,
    RiceBlock,
    LLaVAOneVision1_5_FlashAttention2,
    LLaVAOneVision1_5_TextModel
)
from .divprune import (
    llavaov15_vision_tower_forward_divprune,
    llavaov15_vlmodel_forward_divprune,
)
from .visionzip import (
    llavaov15_vision_flash_attention_forward_visionzip,
    llavaov15_vision_block_forward_visionzip,
    llavaov15_vision_tower_forward_visionzip,
    llavaov15_vlmodel_forward_visionzip
)
from .fastv import (
    llavaov15_flash_attention_forward_fastv,
    llavaov15_language_model_forward_fastv
)


def replace_llavaov15(args, model, method):
    if method == "divprune":
        print("using divprune")
        RiceTransformerPretrainedModel.forward = llavaov15_vision_tower_forward_divprune
        RiceTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        LLaVAOneVision1_5_Model.forward = llavaov15_vlmodel_forward_divprune

    elif method == "visionzip":
        print("using visionzip")
        RiceFlashAttention2.forward = llavaov15_vision_flash_attention_forward_visionzip
        RiceBlock.forward = llavaov15_vision_block_forward_visionzip
        RiceTransformerPretrainedModel.forward = llavaov15_vision_tower_forward_visionzip
        RiceTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        RiceTransformerPretrainedModel.contextual_ratio = getattr(args, 'contextual_ratio', 0.05)
        LLaVAOneVision1_5_Model.forward = llavaov15_vlmodel_forward_visionzip


    elif method == "fastv":
        print("using fastv")
        for name, module in model.named_modules():
            if isinstance(module, LLaVAOneVision1_5_FlashAttention2):
                module.forward = types.MethodType(llavaov15_flash_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, LLaVAOneVision1_5_TextModel):
                module.forward = types.MethodType(llavaov15_language_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)