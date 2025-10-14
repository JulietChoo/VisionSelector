#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="./checkpoints/LLaVA-One-Vision-1.5-8B-adapter-pretrained"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

training_data="../datasets/textvqa_coco4per_ocrvqa_cambrian.jsonl"
output_dir="../output_ckpt/VisionSelector-LLaVA-OV-1.5-8B"


deepspeed src/train/train_sft_visionselector.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path $training_data \
    --image_folder ../datasets \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --freeze_selector False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $output_dir\
    --num_train_epochs 1 \
    --budgets 0.2 \
    --reg_weight_start 0.1 \
    --reg_weight_end 2.0 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((20 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --dataloader_num_workers 4