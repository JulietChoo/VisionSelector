# python3 predict_ov.py \
#     --model_name llava-onevision-1.5 \
#     --method orig \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict_ov.py \
#     --model_name llava-onevision-1.5 \
#     --method fastv \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict_ov.py \
#     --model_name llava-onevision-1.5 \
#     --method visionzip \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

# python3 predict_ov.py \
#     --model_name llava-onevision-1.5 \
#     --method divprune \
#     --budgets 0.2 \
#     --image_path ../docs/logo.png \
#     --question "What is shown in this image?"

python3 predict_ov.py \
    --model_name llava-onevision-1.5 \
    --pretrained ../output_ckpt/VisionSelector-LLaVA-OV-1.5-8B \
    --method selector \
    --budgets 0.2 \
    --image_path ../docs/logo.png \
    --question "What is shown in this image?"