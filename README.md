# UHR-BAT

[Project Page](https://yunkaidang.github.io/bibliography/dang2026uhr-bat/) | [Paper](https://arxiv.org/abs/2604.13565) | [Model](https://huggingface.co/FelixKAI/UHR-BAT) | [SFT Data](https://huggingface.co/datasets/RL-MIND/UHR-BAT-SFT-10K)

UHR-BAT is a budget-aware vision-language framework for ultra-high-resolution remote sensing. It performs query-guided multi-scale token selection and region-faithful compression so kilometer-scale scenes can be processed under a strict context budget.

This repository contains:

- `uhr_bat/`: single-image inference utilities and the `lmms-eval` adapter.
- `with-SAM/`: SAM-guided region partition and multiscale token-mask training/evaluation.
- `with_k-means/`: K-Means-based region partition and attention grouping for training/evaluation.

Both branches are based on LongVA. They expose the same Python package name, `longva`, so install only the branch you are actively running in one environment.

## Installation

The commands below assume CUDA 12.1 and a conda environment named `geo`.

```bash
conda create -n geo python=3.11 -y
conda activate geo

pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

git clone https://github.com/Yunkaidang/UHR-BAT.git
cd UHR-BAT
pip install -r requirements.txt --no-deps
pip install -e . --no-deps
```

Install one LongVA branch:

```bash
cd with_k-means/longva
pip install -e . --no-deps
cd ../..
```

or:

```bash
cd with-SAM/longva
pip install -e . --no-deps
cd ../..
```

For offline machines, download the vision tower once and pass its path with `--vision-tower`:

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 \
  --local-dir checkpoints/clip-vit-large-patch14-336
```

## Data And Checkpoints

The released SFT data is hosted at `RL-MIND/UHR-BAT-SFT-10K`:

```bash
mkdir -p data/UHR-BAT-Data
huggingface-cli download RL-MIND/UHR-BAT-SFT-10K \
  --repo-type dataset \
  --local-dir data/UHR-BAT-Data
```

The released model is hosted at `FelixKAI/UHR-BAT`. Inference can load it directly from Hugging Face, or you can download it locally:

```bash
huggingface-cli download FelixKAI/UHR-BAT --local-dir checkpoints/UHR-BAT
```

Expected local data layout after download:

```text
data/UHR-BAT-Data/
├── ft3_selected_10k.json
├── jpg_images/
├── multiscale_tiles_masks/          # optional, for SAM/mask-based training
├── multiscale_tiles_masks_xlrs/     # optional, for XLRS SAM evaluation
└── multiscale_tiles_masks_mme/      # optional, for MME-RS evaluation
```

## Quick Smoke Test

Run a single-image generation test:

```bash
python -m uhr_bat.infer \
  --image remote_sensing/dota_v2_dota_v2_dota_v2_P8504.png \
  --question "Describe this remote-sensing image briefly." \
  --ckpt FelixKAI/UHR-BAT \
  --image-root data/UHR-BAT-Data/jpg_images \
  --branch kmeans \
  --device cuda:0
```

Equivalent console script:

```bash
uhr-bat-infer \
  --image remote_sensing/dota_v2_dota_v2_dota_v2_P8504.png \
  --question "Describe this remote-sensing image briefly." \
  --ckpt checkpoints/UHR-BAT \
  --image-root data/UHR-BAT-Data/jpg_images \
  --branch kmeans \
  --device cuda:0
```

## Training

The main released SFT recipe uses the SAM/mask branch:

```bash
cd with-SAM/longva

GPU_IDS=0,1,2,3 \
RUN_NAME=uhr-bat-sam \
JSON_PATH=../../data/UHR-BAT-Data/ft3_selected_10k.json \
IMAGE_FOLDER=../../data/UHR-BAT-Data/jpg_images \
MASK_ROOT=../../data/UHR-BAT-Data/multiscale_tiles_masks \
CKPT_PATH=LongVA/LongVA-7B \
LOG_DIR=../../outputs/train_logs \
OUTPUT_DIR=../../outputs/checkpoints/uhr-bat-sam \
bash scripts/ft3_selected.sh
```

Important knobs:

- `TOPK_672`, `TOPK_1344`, `TOPK_2688`, `TOPK_4032`: token budgets for each scale.
- `GPU_IDS`: comma-separated visible GPUs for `torchrun`.
- `CKPT_PATH`: base checkpoint or previous UHR-BAT checkpoint.
- `MASK_ROOT`: multiscale token masks produced by `build_multiscale_token_masks.py`.

K-Means training is available at `with_k-means/longva/scripts/ft3_selected.sh` and does not require `MASK_ROOT`.

## Evaluation Scripts

SAM-based XLRS evaluation:

```bash
cd with-SAM/longva
python -u scripts/eval_xlrs_multiscale_to_json.py \
  --ckpt ../../checkpoints/UHR-BAT \
  --data_json /path/to/xlrs_bench.json \
  --image_root /path/to/image_root \
  --mask_root ../../data/UHR-BAT-Data/multiscale_tiles_masks_xlrs \
  --output_json ../../outputs/xlrs_sam_results.jsonl \
  --multiscale_topk 180,1320,1600,8000 \
  --devices cuda:0,cuda:1
```

K-Means XLRS evaluation:

```bash
cd with_k-means/longva
python -u scripts/eval_xlrs_multiscale_to_json.py \
  --ckpt ../../checkpoints/UHR-BAT \
  --data_json /path/to/xlrs_bench.json \
  --image_root /path/to/image_root \
  --output_json ../../outputs/xlrs_kmeans_results.jsonl \
  --multiscale_topk 80,320,600,2000 \
  --kmeans_num_clusters 600 \
  --kmeans_max_iters 100 \
  --devices cuda:0,cuda:1
```

## lmms-eval

This repository ships an external `lmms-eval` plugin. After installing both repositories, `lmms-eval` will discover the `uhr_bat` model through the `lmms_eval.models` entry point.

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .

cd ../UHR-BAT
pip install -e . --no-deps

lmms-eval --model uhr_bat \
  --model_args pretrained=FelixKAI/UHR-BAT,device_map=auto,multiscale_topk=80:320:600:2000,multiscale_target_sizes=672:1344:2688:4032 \
  --tasks <task_name> \
  --batch_size 1 \
  --limit 10
```

Notes:

- Use `--model uhr_bat`, `--model uhr-bat`, or `--model uhrbat`.
- Use `:` between multiscale values inside `--model_args`; commas are reserved by `lmms-eval` for argument separation.
- The adapter uses the HF remote-code checkpoint and supports image `generate_until` tasks plus loglikelihood-style multiple-choice tasks.
- If you maintain a fork of `lmms-eval` and prefer vendoring the adapter, copy `uhr_bat/lmms_eval_model.py` to `lmms_eval/models/simple/uhr_bat.py` and add `"uhr_bat": "UHRBAT"` to `AVAILABLE_SIMPLE_MODELS`.

## Citation

```bibtex
@inproceedings{dang2026uhrbat,
  title={UHR-BAT: Budget-Aware Token Compression Vision-Language model for Ultra-High-Resolution Remote Sensing},
  author={Dang, Yunkai and Dai, Minxin and Yang, Yuekun and Li, Zhangnan and Li, Wenbin and Miao, Feng and Gao, Yang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgement

- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) for the base multimodal framework.
- [Segment Anything](https://github.com/facebookresearch/segment-anything) for SAM-based region annotations.
- [XLRS-Bench](https://github.com/AI9Stars/XLRS-Bench) for evaluation.
