#!/usr/bin/env bash

# Multiscale masked-token SFT entry for the SAM-guided UHR-BAT branch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export CPATH="$CUDA_HOME/include:${CPATH:-}"
else
  echo "[Warn] CONDA_PREFIX is empty; relying on current CUDA/Python environment."
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# User config
GPU_IDS="${GPU_IDS:-0}"
RUN_NAME="${RUN_NAME:-uhr-bat-sam}"
JSON_PATH="${JSON_PATH:-../../data/UHR-BAT-Data/ft3_selected_10k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../../data/UHR-BAT-Data/jpg_images}"
MASK_ROOT="${MASK_ROOT:-../../data/UHR-BAT-Data/multiscale_tiles_masks}"
CKPT_PATH="${CKPT_PATH:-LongVA/LongVA-7B}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-openai/clip-vit-large-patch14-336}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOG_DIR="${LOG_DIR:-../../outputs/train_logs}"
OUTPUT_DIR="${OUTPUT_DIR:-../../outputs/checkpoints/${RUN_NAME}}"

TOPK_672="${TOPK_672:-80}"
TOPK_1344="${TOPK_1344:-320}"
TOPK_2688="${TOPK_2688:-600}"
TOPK_4032="${TOPK_4032:-2000}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-5000}"
ENABLE_SCALE_POS_RESIDUAL="${ENABLE_SCALE_POS_RESIDUAL:-true}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"

DS_CONFIG="${DS_CONFIG:-scripts/zero2.json}"
if [[ "${DS_CONFIG}" != /* ]]; then
  DS_CONFIG="${REPO_ROOT}/${DS_CONFIG}"
fi
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"

mkdir -p "$LOG_DIR" "$(dirname "$OUTPUT_DIR")"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[Info] Logging stdout/stderr to $LOG_FILE"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

if ! command -v torchrun >/dev/null 2>&1; then
  echo "[Error] torchrun not found. Activate the training environment first." >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/longva/train/train_mem.py" ]]; then
  echo "[Error] train entry not found: ${REPO_ROOT}/longva/train/train_mem.py" >&2
  exit 1
fi
if [[ ! -f "$JSON_PATH" ]]; then
  echo "[Error] JSON_PATH not found: $JSON_PATH" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_FOLDER" ]]; then
  echo "[Error] IMAGE_FOLDER not found: $IMAGE_FOLDER" >&2
  exit 1
fi
if [[ ! -d "$MASK_ROOT" ]]; then
  echo "[Error] MASK_ROOT not found: $MASK_ROOT" >&2
  exit 1
fi
if [[ "${USE_DEEPSPEED}" = "1" && ! -f "$DS_CONFIG" ]]; then
  echo "[Error] DS_CONFIG not found: $DS_CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a _GPU_ARR <<< "$GPU_IDS"
export NUM_GPUS="${#_GPU_ARR[@]}"
export NNODES="${NNODES:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29619}"
export RANK="${RANK:-0}"
export WORLD_SIZE=$((NNODES * NUM_GPUS))

echo "[Info] Using GPUs: ${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"

CMD=(
  torchrun
  --nproc_per_node "${NUM_GPUS}"
  --nnodes "${NNODES}"
  --node_rank "${RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  "${REPO_ROOT}/longva/train/train_mem.py"
  --model_name_or_path "${CKPT_PATH}"
  --mm_tunable_parts "mm_mlp_adapter,mm_language_model,scale_residuals"
  --version "${PROMPT_VERSION}"
  --data_path "${JSON_PATH}"
  --image_folder "${IMAGE_FOLDER}"
  --mask_root "${MASK_ROOT}"
  --multiscale_topk "${TOPK_672},${TOPK_1344},${TOPK_2688},${TOPK_4032}"
  --image_aspect_ratio pad
  --mm_patch_merge_type unires
  --vision_tower "${VISION_MODEL_VERSION}"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --group_by_modality_length True
  --run_name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
  --evaluation_strategy "no"
  --save_strategy "steps"
  --save_steps "${SAVE_STEPS:-100}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --learning_rate "${LEARNING_RATE:-2e-6}"
  --weight_decay 0.
  --warmup_ratio "${WARMUP_RATIO:-0.2}"
  --lr_scheduler_type cosine
  --max_grad_norm "${MAX_GRAD_NORM:-0.5}"
  --logging_steps "${LOGGING_STEPS:-1}"
  --bf16 True
  --tf32 "${TF32:-True}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --gradient_checkpointing True
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}"
  --lazy_preprocess True
  --report_to "${REPORT_TO:-tensorboard}"
  --torch_compile False
  --dataloader_drop_last True
  --enable_scale_pos_residual "${ENABLE_SCALE_POS_RESIDUAL}"
)

if [[ "${USE_DEEPSPEED}" = "1" ]]; then
  CMD+=(--deepspeed "${DS_CONFIG}")
fi

"${CMD[@]}"
