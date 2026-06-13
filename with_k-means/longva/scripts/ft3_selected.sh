#!/usr/bin/env bash

# Multiscale training entry that uses the K-Means attention selection pipeline (segmentation masks are not used).
# Fill in TOPK_* budgets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="$CUDA_HOME/bin:${PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export CPATH="$CUDA_HOME/include:${CPATH:-}"
else
  echo "[Warn] CONDA_PREFIX is empty; relying on current system CUDA/Python environment."
fi
export TOKENIZERS_PARALLELISM=false

# Point PYTHONPATH to this repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=8

export NCCL_DEBUG=INFO

# === User config ===
GPU_IDS="${GPU_IDS:-0}"           # GPUs to use
RUN_NAME="${RUN_NAME:-LongVA-multiscale-kmeans}"
JSON_PATH="${JSON_PATH:-../../data/UHR-BAT-Data/ft3_selected_10k.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../../data/UHR-BAT-Data/jpg_images}"
CKPT_PATH="${CKPT_PATH:-LongVA/LongVA-7B}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"  # flash_attention_2 is less stable on long multiscale seqs
KMEANS_NUM_CLUSTERS="${KMEANS_NUM_CLUSTERS:-3000}"  # fixed K-Means cluster count per scale
KMEANS_MAX_ITERS="${KMEANS_MAX_ITERS:-}"            # optional override for K-Means iteration budget

# === Logging ===
LOG_DIR="${LOG_DIR:-../../outputs/train_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[Info] Logging stdout/stderr to $LOG_FILE"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# k for each scale (tokens to keep per scale)
TOPK_672="${TOPK_672:-80}"
TOPK_1344="${TOPK_1344:-320}"
TOPK_2688="${TOPK_2688:-600}"
TOPK_4032="${TOPK_4032:-2000}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-5000}"
ENABLE_SCALE_POS_RESIDUAL="${ENABLE_SCALE_POS_RESIDUAL:-true}"

VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-openai/clip-vit-large-patch14-336}"
PROMPT_VERSION=qwen_1_5

# DS_CONFIG=${DS_CONFIG:-scripts/zero3_offload.json}
DS_CONFIG=${DS_CONFIG:-scripts/zero2.json}
if [[ "${DS_CONFIG}" != /* ]]; then
  DS_CONFIG="${REPO_ROOT}/${DS_CONFIG}"
fi
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
if [ "${USE_DEEPSPEED}" = "1" ]; then
  DEEPSPEED_ARGS=(--deepspeed "${DS_CONFIG}")
else
  DEEPSPEED_ARGS=()
fi

# Preflight checks
if ! command -v torchrun >/dev/null 2>&1; then
  echo "[Error] torchrun not found in PATH. Please activate your training environment first." >&2
  exit 1
fi
if [ ! -f "${REPO_ROOT}/longva/train/train_mem.py" ]; then
  echo "[Error] train entry not found: ${REPO_ROOT}/longva/train/train_mem.py" >&2
  exit 1
fi
if [ ! -f "${JSON_PATH}" ]; then
  echo "[Error] JSON_PATH not found: ${JSON_PATH}" >&2
  exit 1
fi
if [ ! -d "${IMAGE_FOLDER}" ]; then
  echo "[Error] IMAGE_FOLDER not found: ${IMAGE_FOLDER}" >&2
  exit 1
fi
if [[ "${CKPT_PATH}" == /* || "${CKPT_PATH}" == ./* || "${CKPT_PATH}" == ../* ]]; then
  if [ ! -d "${CKPT_PATH}" ]; then
    echo "[Error] CKPT_PATH not found: ${CKPT_PATH}" >&2
    exit 1
  fi
fi
if [ "${USE_DEEPSPEED}" = "1" ] && [ ! -f "${DS_CONFIG}" ]; then
  echo "[Error] DS_CONFIG not found: ${DS_CONFIG}" >&2
  exit 1
fi

# Optional dataset health check / auto-filter for missing images.
CHECK_DATA_IMAGES="${CHECK_DATA_IMAGES:-1}"
AUTO_FILTER_MISSING_IMAGES="${AUTO_FILTER_MISSING_IMAGES:-1}"
FAIL_ON_MISSING_IMAGES="${FAIL_ON_MISSING_IMAGES:-0}"
if [ "${CHECK_DATA_IMAGES}" = "1" ]; then
  FILTERED_JSON_PATH="${FILTERED_JSON_PATH:-${LOG_DIR}/${RUN_NAME}.filtered.json}"
  CHECK_OUT="$(JSON_PATH="${JSON_PATH}" IMAGE_FOLDER="${IMAGE_FOLDER}" FILTERED_JSON_PATH="${FILTERED_JSON_PATH}" python3 - <<'PY'
import json
import os
from typing import Any, List

json_path = os.environ["JSON_PATH"]
image_folder = os.environ["IMAGE_FOLDER"]
filtered_path = os.environ["FILTERED_JSON_PATH"]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
if not isinstance(data, list):
    raise ValueError(f"Expected list JSON dataset, got: {type(data).__name__}")

def images_of(sample: Any) -> List[str]:
    image = sample.get("image")
    if image is None:
        return []
    if isinstance(image, list):
        return [str(x) for x in image if x is not None]
    return [str(image)]

kept = []
missing = 0
for s in data:
    imgs = images_of(s)
    ok = True
    for rel in imgs:
        p = os.path.join(image_folder, rel)
        if not os.path.isfile(p):
            ok = False
            break
    if ok:
        kept.append(s)
    else:
        missing += 1

print(f"TOTAL={len(data)}")
print(f"MISSING={missing}")
print(f"KEPT={len(kept)}")
if missing > 0:
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    print(f"FILTERED_JSON={filtered_path}")
PY
)"
  echo "${CHECK_OUT}"

  MISSING_COUNT="$(echo "${CHECK_OUT}" | awk -F= '/^MISSING=/{print $2}')"
  FILTERED_JSON="$(echo "${CHECK_OUT}" | awk -F= '/^FILTERED_JSON=/{print $2}')"
  MISSING_COUNT="${MISSING_COUNT:-0}"
  if [ "${MISSING_COUNT}" -gt 0 ]; then
    if [ "${AUTO_FILTER_MISSING_IMAGES}" = "1" ] && [ -n "${FILTERED_JSON}" ]; then
      JSON_PATH="${FILTERED_JSON}"
      echo "[Warn] Found ${MISSING_COUNT} samples with missing images; switched to filtered JSON: ${JSON_PATH}"
    elif [ "${FAIL_ON_MISSING_IMAGES}" = "1" ]; then
      echo "[Error] Found ${MISSING_COUNT} samples with missing images. Set AUTO_FILTER_MISSING_IMAGES=1 or fix dataset." >&2
      exit 1
    else
      echo "[Warn] Found ${MISSING_COUNT} samples with missing images; training may crash on dataloader retries."
    fi
  fi
fi

# GPU setup
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a _GPU_ARR <<< "$GPU_IDS"
export NUM_GPUS="${#_GPU_ARR[@]}"
export NNODES=1
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="${MASTER_PORT:-29619}"
export WORLD_SIZE=$((NNODES * NUM_GPUS))
export RANK=0

echo "[Info] Using GPUs: ${CUDA_VISIBLE_DEVICES} (NUM_GPUS=${NUM_GPUS})"

MULTISCALE_ARGS=(
    --multiscale_topk "${TOPK_672},${TOPK_1344},${TOPK_2688},${TOPK_4032}"
)

KMEANS_ARGS=()
if [[ -n "${KMEANS_NUM_CLUSTERS}" ]]; then
    KMEANS_ARGS+=(--kmeans_num_clusters "${KMEANS_NUM_CLUSTERS}")
fi
if [[ -n "${KMEANS_MAX_ITERS}" ]]; then
    KMEANS_ARGS+=(--kmeans_max_iters "${KMEANS_MAX_ITERS}")
fi

# Torchrun (array form avoids fragile line-continuation parsing issues)
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
  --image_aspect_ratio pad
  --mm_patch_merge_type "unires"
  --vision_tower "${VISION_MODEL_VERSION}"
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --group_by_modality_length True
  --run_name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR:-../../outputs/checkpoints/${RUN_NAME}}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACC_STEPS}"
  --evaluation_strategy "no"
  --save_strategy "steps"
  --save_steps 100
  --save_total_limit 3
  --learning_rate "${LEARNING_RATE:-2e-6}"
  --weight_decay 0.
  --warmup_ratio 0.2
  --lr_scheduler_type "cosine"
  --max_grad_norm 0.5
  --logging_steps 1
  --bf16 True
  --tf32 "${TF32:-True}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --gradient_checkpointing True
  --dataloader_num_workers 0
  --lazy_preprocess True
  --report_to tensorboard
  --torch_compile False
  --dataloader_drop_last True
  --enable_scale_pos_residual "${ENABLE_SCALE_POS_RESIDUAL}"
)

if [ "${USE_DEEPSPEED}" = "1" ]; then
  CMD+=(--deepspeed "${DS_CONFIG}")
fi
CMD+=("${MULTISCALE_ARGS[@]}")
if [ "${#KMEANS_ARGS[@]}" -gt 0 ]; then
  CMD+=("${KMEANS_ARGS[@]}")
fi

"${CMD[@]}"

# Note:
# - train_mem.py consumes --multiscale_topk, calls select_tokens_with_masks (K-Means attention grouping),
#   runs the MLP on selected tokens, then prepares multimodal inputs.
# - The script assumes a single <image> placeholder per sample.
