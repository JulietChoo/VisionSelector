# python3 predict.py \
#     --model_name qwen25-vl \
#     --method fastv \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict.py \
#     --model_name qwen25-vl \
#     --method prumerge+ \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict.py \
#     --model_name qwen25-vl \
#     --method visionzip_official \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"


# python3 predict.py \
#     --model_name qwen25-vl \
#     --method dart \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict.py \
#     --model_name qwen25-vl \
#     --method divprune \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict.py \
#     --pretrained ../output_ckpt/Dynamic-Qwen2.5-VL-7B \
#     --model_name qwen25-vl \
#     --method dynamic \
#     --attn_implementation sdpa \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

python3 predict.py \
    --pretrained ../output_ckpt/VisionSelector-Qwen2.5-VL-7B \
    --model_name qwen25-vl \
    --method selector \
    --budgets 0.2 \
    --image_path ../docs/logo.png \
    --question "What is shown in this image?"