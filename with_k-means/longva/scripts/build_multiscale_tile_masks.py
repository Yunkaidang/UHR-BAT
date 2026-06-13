#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python longva/scripts/build_multiscale_tile_masks.py \
  --data_path /path/to/dataset_manifest.jsonl \
  --image_folder /path/to/image_root \
  --mask_cache_dir /path/to/sam_mask_cache_dir \
  --output_dir /path/to/output_multiscale_tile_masks \
  --scales 672 1344 2688 4032 \
  --tile_size 336 \
  --patch_size 14 \
  --num_workers 28 \
  --sam_repo_path /path/to/segment-anything \
  --overwrite
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _sanitize_cache_name(image_rel_path: str) -> str:
    normalized = os.path.normpath(image_rel_path.strip())
    return normalized.replace("\\", "__").replace("/", "__")


def _iter_image_fields(sample: dict) -> Iterable[str]:
    # Accept both lower/upper-case keys (e.g., "image" vs "Image" in MME JSON)
    image_field = sample.get("image")
    if image_field is None:
        for key, value in sample.items():
            if isinstance(key, str) and key.lower() == "image":
                image_field = value
                break
    if image_field is None:
        return []
    if isinstance(image_field, Sequence) and not isinstance(image_field, (str, bytes)):
        return [str(item) for item in image_field]
    return [str(image_field)]


def _collect_unique_images(data: List[dict]) -> List[str]:
    seen: set = set()
    ordered: List[str] = []
    for sample in data:
        for image_rel in _iter_image_fields(sample):
            if image_rel not in seen:
                seen.add(image_rel)
                ordered.append(image_rel)
    return ordered


def _load_mask_entry(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def _build_label_map_from_annotations(annotations: List[Dict], size: Tuple[int, int], sam_repo_path: str) -> np.ndarray:
    if not annotations:
        return np.zeros((size[1], size[0]), dtype=np.int32)
    if sam_repo_path and sam_repo_path not in sys.path:
        sys.path.append(sam_repo_path)
    from segment_anything.utils.amg import rle_to_mask

    annotations = sorted(annotations, key=lambda ann: ann.get("area", 0), reverse=True)
    W, H = size
    label = np.zeros((H, W), dtype=np.int32)
    for cls_id, ann in enumerate(annotations, start=1):
        mask = rle_to_mask(ann["segmentation"])
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = np.array(Image.fromarray(mask).resize((W, H), resample=Image.NEAREST))
        label[mask.astype(bool)] = cls_id
    return label


def _pad_to_square(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=label.dtype)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    padded[pad_y : pad_y + h, pad_x : pad_x + w] = label
    return padded


def _tile_label(label: np.ndarray, scale: int, tile_size: int = 336, patch_size: int = 14):
    """Pad->resize to scale, then cut into tiles. Return list of (tile_mask, (row_idx, col_idx))."""
    padded = _pad_to_square(label)
    resized = np.array(Image.fromarray(padded).resize((scale, scale), resample=Image.NEAREST))
    tiles = []
    tile_coords = []
    grid_side = scale // tile_size
    tokens_per_side = tile_size // patch_size  # 24 for 336/14
    for row in range(grid_side):
        for col in range(grid_side):
            x0, y0 = col * tile_size, row * tile_size
            x1, y1 = x0 + tile_size, y0 + tile_size
            tile = resized[y0:y1, x0:x1]
            # token majority within 14x14 blocks
            token_mask = np.zeros((tokens_per_side, tokens_per_side), dtype=np.int32)
            for tr in range(tokens_per_side):
                for tc in range(tokens_per_side):
                    px0, py0 = tc * patch_size, tr * patch_size
                    px1, py1 = px0 + patch_size, py0 + patch_size
                    block = tile[py0:py1, px0:px1].reshape(-1)
                    vals, counts = np.unique(block, return_counts=True)
                    token_mask[tr, tc] = int(vals[np.argmax(counts)]) if vals.size > 0 else 0
            tiles.append(torch.from_numpy(token_mask))
            tile_coords.append((row, col))
    return tiles, tile_coords


def _load_dataset(path: Path) -> List[Dict]:
    """Load dataset entries from JSON (list) or JSONL (one JSON object per line)."""
    path = Path(path)
    is_jsonl = path.suffix.lower() in {".jsonl", ".ndjson"}
    with open(path, "r") as f:
        if is_jsonl:
            return [json.loads(line) for line in f if line.strip()]
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def _process_single(image_rel: str, cfg: Dict) -> Tuple[str, bool, str]:
    image_path = Path(cfg["image_folder"]) / image_rel
    mask_path = Path(cfg["mask_cache_dir"]) / f"{_sanitize_cache_name(image_rel)}.pt"
    out_path = Path(cfg["output_dir"]) / f"{_sanitize_cache_name(image_rel)}.pt"

    if not image_path.exists():
        return image_rel, False, f"image missing: {image_path}"
    if not mask_path.exists():
        return image_rel, False, f"mask missing: {mask_path}"
    if out_path.exists() and not cfg["overwrite"]:
        return image_rel, True, "skip"

    try:
        mask_entry = _load_mask_entry(mask_path)
    except Exception as exc:
        return image_rel, False, f"load mask failed: {exc}"

    annotations = mask_entry.get("annotations", [])
    try:
        # Use original image size from disk
        img = Image.open(image_path)
        w, h = img.size
        label_map = _build_label_map_from_annotations(annotations, (w, h), cfg["sam_repo_path"])
    except Exception as exc:
        return image_rel, False, f"build label map failed: {exc}"

    per_scale = {}
    for scale in cfg["scales"]:
        tiles, coords = _tile_label(label_map, scale, tile_size=cfg["tile_size"], patch_size=cfg["patch_size"])
        per_scale[int(scale)] = {"tile_masks": tiles, "tile_coords": coords}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        torch.save({"scales": per_scale}, tmp)
        os.replace(tmp, out_path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return image_rel, False, f"save failed: {exc}"

    return image_rel, True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pixel-level SAM masks to multiscale tile token masks.")
    parser.add_argument("--data_path", required=True, help="JSON/JSONL file with image paths.")
    parser.add_argument("--image_folder", required=True, help="Root folder of images.")
    parser.add_argument("--mask_cache_dir", required=True, help="Directory of pixel-level SAM masks (.pt).")
    parser.add_argument("--output_dir", required=True, help="Directory to save tile token masks.")
    parser.add_argument("--scales", nargs="+", type=int, default=[672, 1344, 2688, 4032], help="Target square scales.")
    parser.add_argument("--tile_size", type=int, default=336, help="Tile size.")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch/token size inside a tile.")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--sam_repo_path", type=str, default="", help="Path to segment-anything repo for RLE decode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = _load_dataset(args.data_path)
    images = _collect_unique_images(data)
    if args.limit is not None:
        images = images[: args.limit]

    cfg = {
        "image_folder": str(Path(args.image_folder).resolve()),
        "mask_cache_dir": str(Path(args.mask_cache_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "scales": args.scales,
        "tile_size": args.tile_size,
        "patch_size": args.patch_size,
        "sam_repo_path": args.sam_repo_path,
        "overwrite": args.overwrite,
    }

    tasks = []
    for rel in images:
        out_path = Path(cfg["output_dir"]) / f"{_sanitize_cache_name(rel)}.pt"
        mask_path = Path(cfg["mask_cache_dir"]) / f"{_sanitize_cache_name(rel)}.pt"
        if not mask_path.exists():
            continue
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append(rel)

    if not tasks:
        print("No images to process.")
        return

    total = len(tasks)
    print(f"Processing {total} images...", flush=True)

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_process_single, rel, cfg) for rel in tasks]
            completed = 0
            for fut in as_completed(futures):
                rel, ok, msg = fut.result()
                completed += 1
                if not ok:
                    print(f"[{completed}/{total}] [Warn] {rel}: {msg}", flush=True)
                else:
                    print(f"[{completed}/{total}] {rel}: {msg}", flush=True)
    else:
        completed = 0
        for rel in tasks:
            _, ok, msg = _process_single(rel, cfg)
            completed += 1
            if not ok:
                print(f"[{completed}/{total}] [Warn] {rel}: {msg}", flush=True)
            else:
                print(f"[{completed}/{total}] {rel}: {msg}", flush=True)


if __name__ == "__main__":
    main()
