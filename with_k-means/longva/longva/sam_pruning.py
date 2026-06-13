import os
import sys
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import torch
from PIL import Image

SAM_GENERATOR = None
SAM_GENERATOR_CFG: Optional[Tuple[str, str, str]] = None
SAM_LOCK = threading.Lock()


def _ensure_repo_on_path(repo_path: Optional[str]) -> None:
    if repo_path and repo_path not in sys.path:
        sys.path.append(repo_path)


def _load_sam_generator(
    checkpoint_path: str,
    model_type: str,
    device: str,
    repo_path: Optional[str],
):
    _ensure_repo_on_path(repo_path)
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model,
        output_mode="uncompressed_rle",
    )
    return generator


def get_sam_generator(
    checkpoint_path: str,
    model_type: str,
    device: str = "cuda",
    repo_path: Optional[str] = None,
):
    global SAM_GENERATOR, SAM_GENERATOR_CFG
    cfg = (checkpoint_path, model_type, device)
    with SAM_LOCK:
        if SAM_GENERATOR is None or SAM_GENERATOR_CFG != cfg:
            SAM_GENERATOR = _load_sam_generator(checkpoint_path, model_type, device, repo_path)
            SAM_GENERATOR_CFG = cfg
    return SAM_GENERATOR


def _convert_patch_box_to_original(
    box: Tuple[int, int, int, int],
    pad_offset: Tuple[int, int],
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x, pad_y = pad_offset

    def _transform(coord: int, pad: int, scale: float, limit: int, fn) -> int:
        if scale == 0:
            return 0
        value = (coord - pad) / scale
        value = fn(value)
        return max(0, min(limit, int(value)))

    orig_x0 = _transform(x0, pad_x, scale_x, original_width, math.floor)
    orig_y0 = _transform(y0, pad_y, scale_y, original_height, math.floor)
    orig_x1 = _transform(x1, pad_x, scale_x, original_width, math.ceil)
    orig_y1 = _transform(y1, pad_y, scale_y, original_height, math.ceil)

    orig_x0 = max(0, min(orig_x0, original_width))
    orig_x1 = max(0, min(orig_x1, original_width))
    orig_y0 = max(0, min(orig_y0, original_height))
    orig_y1 = max(0, min(orig_y1, original_height))
    return orig_x0, orig_y0, orig_x1, orig_y1


def _build_linear_bounds(start: int, end: int, parts: int) -> List[int]:
    """生成从 start 到 end 的均匀切分边界，共 parts 段"""
    if parts <= 0:
        return [start, end]
    total = end - start
    if total <= 0:
        return [start for _ in range(parts + 1)]
    step = total / parts
    bounds = [start]
    for idx in range(1, parts):
        boundary = start + step * idx
        value = int(round(boundary))
        value = max(bounds[-1], min(value, end))
        bounds.append(value)
    bounds.append(end)
    # 确保严格单调递增，避免因为取整出现重复
    for i in range(1, len(bounds)):
        if bounds[i] < bounds[i - 1]:
            bounds[i] = bounds[i - 1]
    bounds[-1] = end
    return bounds


def generate_patch_pruning_info(
    image: Image.Image,
    metadata: Dict,
    checkpoint_path: str,
    model_type: str,
    device: str,
    repo_path: Optional[str] = None,
    vision_patch_size: int = 14,
    token_majority_threshold: float = 0.5,
    sam_image: Optional[Image.Image] = None,
) -> Optional[Dict]:
    if checkpoint_path is None:
        raise ValueError("SAM pruning requested but no checkpoint path provided.")

    generator = get_sam_generator(checkpoint_path, model_type, device, repo_path)

    image_for_sam = sam_image if sam_image is not None else image
    image_np = np.array(image_for_sam.convert("RGB"))
    annotations = generator.generate(image_np)
    if len(annotations) == 0:
        patch_count = len(metadata.get("patch_coords", []))
        return {
            "patch_classes": [0 for _ in range(patch_count)],
            "mask_order": [],
            "mask_areas": [],
        }

    annotations = sorted(annotations, key=lambda ann: ann.get("area", 0), reverse=True)

    from segment_anything.utils.amg import rle_to_mask

    pad_offset = metadata["pad_offset"]
    scale_x = metadata["scale_x"]
    scale_y = metadata["scale_y"]
    orig_w, orig_h = metadata["original_size"]
    patch_size = metadata["patch_size"]

    mask_scale_x = image_for_sam.width / max(1, orig_w)
    mask_scale_y = image_for_sam.height / max(1, orig_h)
    inv_scale_area = 1.0
    if mask_scale_x > 0 and mask_scale_y > 0:
        inv_scale_area = 1.0 / (mask_scale_x * mask_scale_y)

    patch_coords = metadata.get("patch_coords", [])
    patch_boxes: List[Tuple[int, int, int, int]] = []
    patch_areas: List[int] = []
    for box in patch_coords:
        orig_box = _convert_patch_box_to_original(box, pad_offset, scale_x, scale_y, orig_w, orig_h)
        x0, y0, x1, y1 = orig_box
        patch_boxes.append(orig_box)
        patch_areas.append(max(0, x1 - x0) * max(0, y1 - y0))

    patch_classes = [0 for _ in patch_boxes]
    patch_coverages = [0.0 for _ in patch_boxes]
    mask_order = list(range(1, len(annotations) + 1))
    mask_areas = [ann.get("area", 0) for ann in annotations]

    tokens_per_side = max(1, patch_size // max(1, vision_patch_size))
    tokens_per_patch = tokens_per_side * tokens_per_side
    total_tokens = len(patch_boxes) * tokens_per_patch
    token_classes = [0 for _ in range(total_tokens)]
    token_coverages = [0.0 for _ in range(total_tokens)]

    patch_token_bounds: List[Tuple[List[int], List[int]]] = []
    for box in patch_boxes:
        x0, y0, x1, y1 = box
        x_bounds = _build_linear_bounds(x0, x1, tokens_per_side)
        y_bounds = _build_linear_bounds(y0, y1, tokens_per_side)
        patch_token_bounds.append((x_bounds, y_bounds))

    def _map_range(start: int, end: int, scale: float, limit: int) -> Tuple[int, int]:
        if scale <= 0:
            return 0, 0
        mapped_start = int(math.floor(start * scale))
        mapped_end = int(math.ceil(end * scale))
        mapped_start = max(0, min(mapped_start, limit))
        mapped_end = max(mapped_start, min(mapped_end, limit))
        return mapped_start, mapped_end

    for mask_idx, ann in enumerate(annotations):
        mask_data = rle_to_mask(ann["segmentation"])
        if isinstance(mask_data, torch.Tensor):
            mask_arr = mask_data.cpu().numpy()
        else:
            mask_arr = np.asarray(mask_data)
        mask_h, mask_w = mask_arr.shape[:2]
        for patch_idx, (box, area) in enumerate(zip(patch_boxes, patch_areas)):
            if area == 0:
                continue
            x0, y0, x1, y1 = box
            if x0 >= x1 or y0 >= y1:
                continue
            mapped_y0, mapped_y1 = _map_range(y0, y1, mask_scale_y, mask_h)
            mapped_x0, mapped_x1 = _map_range(x0, x1, mask_scale_x, mask_w)
            if mapped_y1 <= mapped_y0 or mapped_x1 <= mapped_x0:
                continue
            base_index = patch_idx * tokens_per_patch
            x_bounds, y_bounds = patch_token_bounds[patch_idx]
            for token_row in range(tokens_per_side):
                ty0, ty1 = y_bounds[token_row], y_bounds[token_row + 1]
                if ty1 <= ty0:
                    continue
                for token_col in range(tokens_per_side):
                    tx0, tx1 = x_bounds[token_col], x_bounds[token_col + 1]
                    if tx1 <= tx0:
                        continue
                    token_idx = base_index + token_row * tokens_per_side + token_col
                    token_area = (ty1 - ty0) * (tx1 - tx0)
                    if token_area <= 0:
                        continue
                    my0, my1 = _map_range(ty0, ty1, mask_scale_y, mask_h)
                    mx0, mx1 = _map_range(tx0, tx1, mask_scale_x, mask_w)
                    if my1 <= my0 or mx1 <= mx0:
                        continue
                    coverage = mask_arr[my0:my1, mx0:mx1].sum() * inv_scale_area
                    if coverage <= 0:
                        continue
                    coverage_ratio = float(coverage) / float(token_area)
                    if coverage_ratio > token_majority_threshold and coverage_ratio > token_coverages[token_idx]:
                        token_coverages[token_idx] = coverage_ratio
                        token_classes[token_idx] = mask_idx + 1

            coverage = mask_arr[mapped_y0:mapped_y1, mapped_x0:mapped_x1].sum() * inv_scale_area
            if coverage <= 0:
                continue
            ratio = float(coverage) / float(area)
            if ratio > 0.5 and ratio > patch_coverages[patch_idx]:
                patch_coverages[patch_idx] = ratio
                patch_classes[patch_idx] = mask_idx + 1
        del mask_arr

    # 若 patch 层没有覆盖，仍然依据 token 结果选出多数
    if tokens_per_patch > 0:
        for patch_idx in range(len(patch_boxes)):
            start = patch_idx * tokens_per_patch
            end = start + tokens_per_patch
            token_slice = token_classes[start:end]
            counts: Dict[int, int] = {}
            for cls_id in token_slice:
                if cls_id == 0:
                    continue
                counts[cls_id] = counts.get(cls_id, 0) + 1
            if counts:
                patch_classes[patch_idx] = max(counts.items(), key=lambda x: x[1])[0]

    return {
        "patch_classes": patch_classes,
        "mask_order": mask_order,
        "mask_areas": mask_areas,
        "token_classes": token_classes,
        "tokens_per_patch": tokens_per_patch,
        "tokens_per_side": tokens_per_side,
        "token_coverages": token_coverages,
    }
