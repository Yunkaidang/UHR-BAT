#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



python build_multiscale_token_masks.py \
  --data_path /path/to/train.json \
  --image_folder /path/to/images \
  --mask_cache_dir /path/to/pixel_masks \
  --output_dir /path/to/token_masks \
  --scales 672 1344 2688 4032 \
  --vision_patch_size 14 \
  --num_workers 8


"""

import argparse
import json
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
    image_field = sample.get("image")
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

    # 排序保证前面的标签优先覆盖
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


def _pad_and_resize_label(label: np.ndarray, target_size: int) -> np.ndarray:
    h, w = label.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=label.dtype)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    padded[pad_y : pad_y + h, pad_x : pad_x + w] = label
    resized = np.array(Image.fromarray(padded).resize((target_size, target_size), resample=Image.NEAREST))
    return resized


def _label_to_tokens(label_resized: np.ndarray, patch_size: int) -> torch.Tensor:
    if label_resized.shape[0] % patch_size != 0 or label_resized.shape[1] % patch_size != 0:
        raise ValueError(f"Label size {label_resized.shape} not divisible by patch_size={patch_size}")
    h_tokens = label_resized.shape[0] // patch_size
    w_tokens = label_resized.shape[1] // patch_size
    tokens = np.zeros((h_tokens, w_tokens), dtype=np.int32)
    for i in range(h_tokens):
        for j in range(w_tokens):
            block = label_resized[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            flat = block.reshape(-1)
            if flat.size == 0:
                tokens[i, j] = 0
                continue
            vals, counts = np.unique(flat, return_counts=True)
            tokens[i, j] = int(vals[np.argmax(counts)])
    return torch.from_numpy(tokens)


def _process_single(
    image_rel: str,
    cfg: Dict,
) -> Tuple[str, bool, str]:
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
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return image_rel, False, f"open image failed: {exc}"

    try:
        mask_entry = _load_mask_entry(mask_path)
    except Exception as exc:
        return image_rel, False, f"load mask failed: {exc}"

    annotations = mask_entry.get("annotations", [])
    label_map = _build_label_map_from_annotations(annotations, img.size, cfg["sam_repo_path"])

    per_scale = {}
    for scale in cfg["scales"]:
        resized = _pad_and_resize_label(label_map, scale)
        tokens = _label_to_tokens(resized, cfg["vision_patch_size"])
        per_scale[int(scale)] = tokens

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
    parser = argparse.ArgumentParser(description="Convert pixel-level SAM masks to multiscale token masks.")
    parser.add_argument("--data_path", required=True, help="JSON file with image paths.")
    parser.add_argument("--image_folder", required=True, help="Root folder of images.")
    parser.add_argument("--mask_cache_dir", required=True, help="Directory of pixel-level SAM masks (.pt).")
    parser.add_argument("--output_dir", required=True, help="Directory to save token masks.")
    parser.add_argument("--scales", nargs="+", type=int, default=[672, 1344, 2688, 4032], help="Target square scales.")
    parser.add_argument("--vision_patch_size", type=int, default=14, help="Patch size (CLIP 14).")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--sam_repo_path", type=str, default="", help="Path to segment-anything repo for RLE decode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.data_path, "r"))
    images = _collect_unique_images(data)
    if args.limit is not None:
        images = images[: args.limit]

    cfg = {
        "image_folder": str(Path(args.image_folder).resolve()),
        "mask_cache_dir": str(Path(args.mask_cache_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "scales": args.scales,
        "vision_patch_size": args.vision_patch_size,
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

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_process_single, rel, cfg) for rel in tasks]
            for fut in as_completed(futures):
                rel, ok, msg = fut.result()
                if not ok:
                    print(f"[Warn] {rel}: {msg}")
    else:
        for rel in tasks:
            _, ok, msg = _process_single(rel, cfg)
            if not ok:
                print(f"[Warn] {rel}: {msg}")


if __name__ == "__main__":
    main()
