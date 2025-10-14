# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm, Qwen2_5_VLVisionFlashAttention2, Qwen2_5_VLVisionBlock
import torch.nn as nn
from compression_method.selector_scorer import TransformerScorer
from compression_method.selector_model import (
   qwen25vl_vision_tower_forward_selector,
   qwen25vl_generation_forward_selector 
)
import types
import random
import numpy as np

local_rank = None

class ScheduledWeightTrainer(Trainer):
    def __init__(self, *args, reg_weight_start=0.1, reg_weight_end=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_weight_start = reg_weight_start
        self.reg_weight_end = reg_weight_end

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Overrides compute_loss to dynamically calculate regularization_weight.
        """
        total_steps = self.state.max_steps
        current_step = self.state.global_step

        if total_steps > 0:
            # Use min to ensure progress does not exceed 1.0
            progress = min(current_step / total_steps, 1.0)
            current_weight = self.reg_weight_start + (self.reg_weight_end - self.reg_weight_start) * progress
        else:
            # Use the starting weight if total_steps is not yet computed (value is -1)
            current_weight = self.reg_weight_start

        # Set the calculated weight on the actual model
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.regularization_weight = current_weight

        # Log the weight
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            # Print only on the main process to avoid duplicates
            if self.is_world_process_zero():
                print(f"\n[Step {self.state.global_step}] Set regularization_weight to: {current_weight:.4f}")

        # Call the parent's compute_loss method
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False
    # -------------------------add compressor tuning---------------------------------
    if model_args.tune_compressor:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.importance_scorer.named_parameters():
            p.requires_grad = False
    # -------------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print('seed:', training_args.seed)
    print('data_seed:', training_args.data_seed)
    set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError("Model not currently supported")
        
    # ----------------------------add compressor setup-------------------------------
    model.visual.budgets = model_args.budget
    model.visual.forward = types.MethodType(qwen25vl_vision_tower_forward_selector, model.visual)
    if "3b" in model_args.model_name_or_path.lower():
        print("3b")
        model.visual.importance_scorer = TransformerScorer(in_features=2048,hidden_dim=1024)
    elif "7b" in model_args.model_name_or_path.lower():   
        print("7b")
        model.visual.importance_scorer = TransformerScorer(in_features=3584,hidden_dim=1792)
    else:
        raise ValueError("Model not currently supported")
    model.forward = types.MethodType(qwen25vl_generation_forward_selector, model)
    # -------------------------------------------------------------------------------

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False


    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # --- print trainable parameters ---
    if local_rank == 0: 
        print("="*80)
        print("Printing trainable parameters...")
        trainable_param_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        for name in trainable_param_names:
            print(f"- {name}")
        print("="*80)
    # -----------------------------------

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = ScheduledWeightTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reg_weight_start=training_args.reg_weight_start,  # Set starting weight
        reg_weight_end=training_args.reg_weight_end,    # Set ending weight
        **data_module
    )

    # trainer = Trainer(
    #     model=model, processing_class=tokenizer, args=training_args, **data_module
    # )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # After training, copy preprocessor_config.json and chat_template.json
    if local_rank == 0:
        source_dir = pathlib.Path(model_args.model_name_or_path)
        dest_dir = pathlib.Path(training_args.output_dir)
        
        preprocessor_file = "preprocessor_config.json"
        chat_template_file = "chat_template.json"
        
        # Copy preprocessor_config.json
        source_preprocessor_path = source_dir / preprocessor_file
        dest_preprocessor_path = dest_dir / preprocessor_file
        if source_preprocessor_path.exists():
            shutil.copy(source_preprocessor_path, dest_preprocessor_path)
            rank0_print(f"Copied {source_preprocessor_path} to {dest_preprocessor_path}")
        else:
            rank0_print(f"Warning: {source_preprocessor_path} not found.")

        # Copy chat_template.json
        source_chat_template_path = source_dir / chat_template_file
        dest_chat_template_path = dest_dir / chat_template_file
        if source_chat_template_path.exists():
            shutil.copy(source_chat_template_path, dest_chat_template_path)
            rank0_print(f"Copied {source_chat_template_path} to {dest_chat_template_path}")
        else:
            rank0_print(f"Warning: {source_chat_template_path} not found.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
