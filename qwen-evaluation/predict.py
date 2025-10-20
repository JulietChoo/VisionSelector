# inference.py


import copy
import json
import base64
import logging
import argparse
import torch
import os
from io import BytesIO
from qwen25vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen25vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import requests
import transformers
import numpy as np
import random

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="A simple inference script to test token prune and kv cache compression methods.")
    # settings for path/basic
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--image_path", type=str, default="../docs/logo.png")
    parser.add_argument("--question", type=str, default="What is shown in this image?",help="the question to ask the model")

    # settings for model configuration
    parser.add_argument('--pretrained', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help='Pretrained model path or identifier.')
    parser.add_argument('--model_name', type=str, default='qwen25-vl', help='Model name. such as Qwen2-VL-7B-Instruct.')
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="")
    parser.add_argument("--temperature", type=float, default=0, help="")
    parser.add_argument("--top_p", type=float, default=None, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="")
    parser.add_argument("--multimodal", type=bool, default=True, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="")
    # for qwen25-vl
    parser.add_argument("--max_pixels", type=int, default=2800, help="the max pixels of the image for qwen25-vl")
    parser.add_argument("--min_pixels", type=int, default=400, help="the min pixels of the image for qwen25-vl")
    
    # settings for token pruning method
    parser.add_argument('--method', type=str, choices=['orig', # original model
                                                       'fastv', # token prune method
                                                       'visionzip_official',
                                                       'prumerge+',
                                                       'dart',
                                                       'divprune',
                                                       'dynamic',
                                                       'selector'], help='Token pruning method to use.')
    parser.add_argument("--budgets", type=float, default=0.2, help="budgets of Token Pruning")

    
    return parser.parse_args()

def replace_layers(args,model):
    from token_compression.monkeypatch import replace_qwen25vl
    
    if "qwen25-vl" in args.model_name.lower():
        replace_qwen25vl(args,model,args.method.lower())
    else:
        raise ValueError(f"Model name {args.model_name} not supported")


def run_inference(args):

    if "qwen25-vl" in args.model_name.lower():
        run_inference_qwen25vl(args)
    else:
        raise ValueError(f"Model name {args.model_name} not supported")
    
def load_model_qwen25vl(args, pretrained, model_name):
    from token_compression.visionzip_official import Qwen2_5_VLForConditionalGeneration_VisionZip
    from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
    from token_compression.dynamic_model import Qwen2_5_VLForConditionalGeneration_Dynamic

    model_args = {
        "attn_implementation": args.attn_implementation,
        "device_map": args.device_map, 
        "torch_dtype": args.torch_dtype,
    }

    if args.method.lower() == 'visionzip_official':
        print('using visionzip official')
        model = Qwen2_5_VLForConditionalGeneration_VisionZip.from_pretrained(pretrained, **model_args).eval() 
        model.budget = args.budgets
    elif args.method.lower() == 'selector':
        print('using selector')
        model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(pretrained, **model_args).eval()
        model.visual.budgets = args.budgets
    elif args.method.lower() == 'dynamic':
        print('using dynamic')
        model = Qwen2_5_VLForConditionalGeneration_Dynamic.from_pretrained(pretrained, **model_args).eval()
        model.model.budgets = args.budgets
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_args).eval() 

    qwen25vl_processor = Qwen2_5_VLProcessor.from_pretrained(pretrained, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    qwen25vl_tokenizer = AutoTokenizer.from_pretrained(pretrained)

    return model, qwen25vl_processor, qwen25vl_tokenizer


def run_inference_qwen25vl(args):

    model, qwen25vl_processor, qwen25vl_tokenizer = load_model_qwen25vl(args, args.pretrained, args.model_name)
    replace_layers(args,model)

    messages = []
    processed_visuals = []
    context = args.question
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    # process image
    visual = Image.open(args.image_path)
    base64_image = visual.convert("RGB")
    buffer = BytesIO()
    base64_image.save(buffer, format="JPEG")
    base64_bytes = base64.b64encode(buffer.getvalue())
    base64_string = base64_bytes.decode("utf-8")

    # message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": args.max_pixels, "min_pixels": args.min_pixels}, {"type": "text", "text": context}]})
    message.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": context}]})
    messages.append(message)
    texts = [qwen25vl_processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen25vl_processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    if args.device_map == "auto":
        inputs = inputs.to("cuda")
    else:
        inputs = inputs.to(args.device)
    pad_token_id = qwen25vl_tokenizer.pad_token_id

    # generate
    cont = model.generate(
        **inputs,
        eos_token_id= qwen25vl_tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=args.use_cache,
    )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
    answers = qwen25vl_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print answer
    print(answers[0])


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)