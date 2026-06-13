# UHR-BAT: Budget-Aware Token Compression Vision-Language Model for Ultra-High-Resolution Remote Sensing

[![Paper](https://img.shields.io/badge/arXiv-2604.13565-b31b1b.svg)](https://arxiv.org/abs/2604.13565)
[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)]()
[![Model](https://img.shields.io/badge/Model-RL--MIND%2FUHR--BAT-yellow.svg)](https://huggingface.co/RL-MIND/UHR-BAT)
[![SFT Data](https://img.shields.io/badge/SFT%20Data-RL--MIND%2FUHR--BAT--SFT--10K-green.svg)](https://huggingface.co/datasets/RL-MIND/UHR-BAT-SFT-10K)
[![Eval Data](https://img.shields.io/badge/Eval%20Data-RL--MIND%2FXHRBench-orange.svg)](https://huggingface.co/datasets/RL-MIND/XHRBench)

[Project Page](https://yunkaidang.github.io/bibliography/dang2026uhr-bat/) | [Paper](https://arxiv.org/abs/2604.13565) | [Model](https://huggingface.co/RL-MIND/UHR-BAT) | [SFT Data](https://huggingface.co/datasets/RL-MIND/UHR-BAT-SFT-10K) | [Eval Data](https://huggingface.co/datasets/RL-MIND/XHRBench)

UHR-BAT is a budget-aware vision-language framework for ultra-high-resolution remote sensing. It targets kilometer-scale scenes where query-critical evidence may occupy only a few pixels. Instead of relying on direct downsampling, dense tiling, or generic global pruning, UHR-BAT performs query-guided multi-scale token selection and region-faithful compression so large remote-sensing images can be processed under a strict context budget.

## News

- **2026**: UHR-BAT has been accepted by **ICML 2026**.
- **2026**: The source code, pretrained model, supervised fine-tuning data, and XHRBench evaluation data are released.

## Introduction

Ultra-high-resolution remote-sensing images contain rich visual details but introduce severe computational challenges for vision-language models. Directly resizing such images can erase small but decisive objects, while dense tiling and global token pruning often produce unpredictable compute or discard query-relevant regions.

UHR-BAT introduces a budget-aware token compression strategy for efficient and effective understanding of ultra-high-resolution remote-sensing imagery. It allocates visual token budgets according to the current instruction, preserves informative regional evidence, and merges redundant background tokens into compact representatives.

## Highlights

- Designed for ultra-high-resolution remote-sensing image understanding.
- Query-guided token compression allocates token budgets according to the current instruction.
- Multi-scale input preserves both global scene context and fine-grained local evidence.
- Region-faithful preserve-and-merge keeps task-relevant regional tokens while reducing redundancy.
- Efficient UHR understanding under memory, latency, and context-budget constraints.

## Main Results

The project page and model card report strong ultra-high-resolution remote-sensing results under strict token budgets:

- XLRS-Bench: 44.0 weighted average accuracy.
- MMERealworld-RS: 33.33 mean score.
- RSHR-Bench: 29.2 on Perception and 45.0 on Reasoning.

## Repository Contents

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

## Models And Datasets

The released model, SFT data, and evaluation benchmark are hosted under `RL-MIND` on Hugging Face.

| Asset | Hugging Face repository | Purpose |
| --- | --- | --- |
| UHR-BAT model | [`RL-MIND/UHR-BAT`](https://huggingface.co/RL-MIND/UHR-BAT) | Full pretrained UHR-BAT checkpoint with remote-code wrappers. |
| SFT data | [`RL-MIND/UHR-BAT-SFT-10K`](https://huggingface.co/datasets/RL-MIND/UHR-BAT-SFT-10K) | Supervised fine-tuning data for UHR-BAT. |
| Evaluation data | [`RL-MIND/XHRBench`](https://huggingface.co/datasets/RL-MIND/XHRBench) | Ultra-high-resolution remote-sensing evaluation benchmark. |

Download the released assets:

```bash
mkdir -p checkpoints data

huggingface-cli download RL-MIND/UHR-BAT \
  --local-dir checkpoints/UHR-BAT

huggingface-cli download RL-MIND/UHR-BAT-SFT-10K \
  --repo-type dataset \
  --local-dir data/UHR-BAT-SFT-10K

huggingface-cli download RL-MIND/XHRBench \
  --repo-type dataset \
  --local-dir data/XHRBench
```

Prepare the released SFT annotations for training:

```bash
python scripts/prepare_uhrbat_sft.py \
  --metadata data/UHR-BAT-SFT-10K/train/metadata.parquet \
  --output data/UHR-BAT-SFT-10K/ft3_selected_10k.json
```

Notes on local layouts:

- `RL-MIND/UHR-BAT-SFT-10K` is packaged as `train/metadata.parquet` and `train/images/`. Convert `metadata.parquet` to LongVA-style JSON before training, then use `IMAGE_FOLDER=data/UHR-BAT-SFT-10K/train/images`.
- The released training scripts expect a LongVA-style JSON file plus an image folder through `JSON_PATH` and `IMAGE_FOLDER`.
- `RL-MIND/XHRBench` includes `dataset.json` and an `images/` folder. Because `dataset.json` stores paths such as `images/xxx.png`, use `--image_root data/XHRBench`.
- Model weights, datasets, checkpoints, and output folders are ignored by `.gitignore` and should not be committed to this repository.

## Quick Smoke Test

Run a single-image generation test with the released Hugging Face checkpoint:

```bash
python -m uhr_bat.infer \
  --image /path/to/remote_sensing_image.png \
  --question "Describe this remote-sensing image briefly." \
  --ckpt RL-MIND/UHR-BAT \
  --branch kmeans \
  --device cuda:0
```

Equivalent console script:

```bash
uhr-bat-infer \
  --image /path/to/remote_sensing_image.png \
  --question "Describe this remote-sensing image briefly." \
  --ckpt RL-MIND/UHR-BAT \
  --branch kmeans \
  --device cuda:0
```

For a relative image path under XHRBench, pass the image root explicitly:

```bash
uhr-bat-infer \
  --image <relative_image_path_from_dataset_json> \
  --image-root data/XHRBench \
  --question "Describe this remote-sensing image briefly." \
  --ckpt RL-MIND/UHR-BAT \
  --branch kmeans \
  --device cuda:0
```

## Training

The main released SFT recipe uses the SAM/mask branch. The example below assumes you have prepared a LongVA-style JSON annotation file, an image folder, and multiscale token masks:

```bash
cd with-SAM/longva

GPU_IDS=0,1,2,3 \
RUN_NAME=uhr-bat-sam \
JSON_PATH=../../data/UHR-BAT-SFT-10K/ft3_selected_10k.json \
IMAGE_FOLDER=../../data/UHR-BAT-SFT-10K/train/images \
MASK_ROOT=/path/to/multiscale_tiles_masks \
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

The K-Means branch can evaluate XHRBench without precomputed masks:

```bash
cd with_k-means/longva
python -u scripts/eval_xlrs_multiscale_to_json.py \
  --ckpt RL-MIND/UHR-BAT \
  --data_json ../../data/XHRBench/dataset.json \
  --image_root ../../data/XHRBench \
  --output_json ../../outputs/xhrbench_kmeans_results.jsonl \
  --multiscale_topk 80,320,600,2000 \
  --kmeans_num_clusters 600 \
  --kmeans_max_iters 100 \
  --devices cuda:0,cuda:1
```

SAM-based evaluation requires precomputed multiscale token masks for the evaluation images:

```bash
cd with-SAM/longva
python -u scripts/eval_xlrs_multiscale_to_json.py \
  --ckpt RL-MIND/UHR-BAT \
  --data_json ../../data/XHRBench/dataset.json \
  --image_root ../../data/XHRBench \
  --mask_root /path/to/xhrbench_multiscale_tile_masks \
  --output_json ../../outputs/xhrbench_sam_results.jsonl \
  --multiscale_topk 180,1320,1600,8000 \
  --devices cuda:0,cuda:1
```

The script name keeps the original `xlrs` convention, but the loader accepts XLRS/MME-style JSON lists and can be pointed at XHRBench's `dataset.json`.

## lmms-eval

This repository ships an external `lmms-eval` plugin. After installing both repositories, `lmms-eval` will discover the `uhr_bat` model through the `lmms_eval.models` entry point.

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .

cd ../UHR-BAT
pip install -e . --no-deps

lmms-eval --model uhr_bat \
  --model_args pretrained=RL-MIND/UHR-BAT,device_map=auto,multiscale_topk=80:320:600:2000,multiscale_target_sizes=672:1344:2688:4032 \
  --tasks <task_name> \
  --batch_size 1 \
  --limit 10
```

Notes:

- Use `--model uhr_bat`, `--model uhr-bat`, or `--model uhrbat`.
- Use `:` between multiscale values inside `--model_args`; commas are reserved by `lmms-eval` for argument separation.
- The adapter uses the Hugging Face remote-code checkpoint and supports image `generate_until` tasks plus loglikelihood-style multiple-choice tasks.
- If you maintain a fork of `lmms-eval` and prefer vendoring the adapter, copy `uhr_bat/lmms_eval_model.py` to `lmms_eval/models/simple/uhr_bat.py` and add `"uhr_bat": "UHRBAT"` to `AVAILABLE_SIMPLE_MODELS`.

## Citation

If you find this work useful, please consider citing our paper:

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
- [XLRS-Bench](https://github.com/AI9Stars/XLRS-Bench) and [XHRBench](https://huggingface.co/datasets/RL-MIND/XHRBench) for ultra-high-resolution remote-sensing evaluation.
