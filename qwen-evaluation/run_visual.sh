# save result path
ROOT_DIR="./result/visualizations"
mkdir -p "$ROOT_DIR"
NUM_PROCESSES=8
export OPENAI_API_URL=""
export OPENAI_API_KEY=""
set -x


BASE_COMMAND="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_5_vl_with_token_compression_visual \
    --batch_size 1 \
    --log_samples"

# first for method name ; second for FILENAME ; third for additional args
METHODS=(
    # Qwen2-5-VL token compression method for example   
    "divprune visualizations attn_implementation="flash_attention_2""
    "visionzip visualizations contextual_ratio=0.05,attn_implementation="flash_attention_2""
    # "selector visualizations attn_implementation="flash_attention_2""
)




# budgets
BUDGETS=(0.2)

# model path
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="qwen25vl_7b"
# MODEL_PATH="../output_ckpt/VisionSelector-Qwen2.5-VL-7B"
# MODEL_NAME="VisionSelector-Qwen2.5-VL-7B"

TASKS=("textvqa_val" "ocrbench")

for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

        for BUDGET in "${BUDGETS[@]}"; do
            OUTPUT_PATH="$ROOT_DIR/${TASK}/${BUDGET}/${MODEL_NAME}_${METHOD}_${BUDGET}_${FILENAME}"


            if [ -d "$OUTPUT_PATH" ]; then
                echo "folder exists, skip."
                continue
            fi

            mkdir -p "$OUTPUT_PATH"

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},${ADDITIONAL_ARGS}"

            COMMAND="${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args \"${MODEL_ARGS}\""
            
            eval "${COMMAND}"

        done
    done
done
