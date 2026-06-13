#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the Remote Sensing split of MME-RealWorld-Lite with multiscale mask + attention pruning.

参考 longva/scripts/eval_xlrs_multiscale_to_json.py，实现逻辑：
- 仅过滤 JSON 中 Subtask == "Remote Sensing" 的样本；
- 使用多尺度 tiles + 掩码进行推理，实时写入 JSONL；
- 结束后按 Task 与 Category 统计准确率。

示例：
python -u longva/scripts/eval_mme_remote_sensing_multiscale.py \
  --ckpt /path/to/checkpoint_dir \
  --data_json /path/to/mme_remote_sensing_dataset.json \
  --image_root /path/to/mme_images_root \
  --mask_root /path/to/multiscale_tile_mask_dir \
  --output_json /path/to/output_results.jsonl \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --multiscale_topk 80,320,600,2000 \
  --max_samples 0


  python -u longva/scripts/eval_xlrs_multiscale_to_json.py \
    --ckpt /path/to/checkpoint_dir \
    --data_json /path/to/mme_dataset.json \
    --image_root /path/to/mme_images_root \
    --mask_root /path/to/multiscale_tile_mask_dir \
    --output_json /path/to/output_results.jsonl \
    --multiscale_topk 80,320,600,2000 \
    --max_samples 0 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

# Allow huge remote-sensing images
Image.MAX_IMAGE_PIXELS = None

# Add repo root to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from longva.conversation import conv_templates  # noqa: E402
from longva.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from longva.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    load_multiscale_tile_masks,
    process_images,
    split_image_to_multiscale_tiles,
    tokenizer_image_token,
)
from longva.model.builder import load_pretrained_model  # noqa: E402

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]
OPTION_PREFIX_RE = re.compile(r"^\s*\(?([A-Z])\)?[).:\s-]*", re.IGNORECASE)
LEADING_IMAGE_PATTERN = re.compile(r"^((?:\s*<image>\s*)+)", re.IGNORECASE)


def _sanitize_cache_name(image_rel_path: str) -> str:
    normalized = os.path.normpath(image_rel_path.strip())
    return normalized.replace("\\", "__").replace("/", "__")


def _strip_leading_image_tokens(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    match = LEADING_IMAGE_PATTERN.match(text)
    if not match:
        return text
    remainder = text[match.end() :]
    remainder = remainder.lstrip("\r\n")
    removed = match.group(1)
    if "\n" in removed and remainder and not remainder.startswith("\n"):
        remainder = "\n" + remainder
    return remainder


def _parse_int_list(val: str) -> List[int]:
    return [int(x) for x in str(val).split(",") if str(x).strip()]


def _load_completed_ids(out_path: Path) -> set:
    ids = set()
    if not out_path.exists():
        return ids
    try:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sid = obj.get("id")
                if sid is not None:
                    ids.add(sid)
    except Exception:
        pass
    return ids


def _append_line(lock: Any, out_path: Path, line: str) -> None:
    if lock:
        try:
            lock.acquire()
        except Exception:
            lock = None
    try:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line)
    finally:
        if lock:
            try:
                lock.release()
            except Exception:
                pass


def _safe_batch_decode(tokenizer, sequences, **decode_kwargs) -> List[str]:
    try:
        seq_tensor = torch.as_tensor(sequences, dtype=torch.long)
    except Exception:
        return tokenizer.batch_decode(sequences, **decode_kwargs)
    replace_id = tokenizer.pad_token_id
    if replace_id is None:
        replace_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    seq_tensor = seq_tensor.clone()
    seq_tensor = seq_tensor.masked_fill(seq_tensor == IMAGE_TOKEN_INDEX, replace_id)
    seq_tensor = seq_tensor.masked_fill(seq_tensor < 0, replace_id)
    if vocab_size:
        seq_tensor = seq_tensor.masked_fill(seq_tensor >= vocab_size, replace_id)
    return tokenizer.batch_decode(seq_tensor, **decode_kwargs)


def decode_chatml_response(tokenizer, sequences: torch.LongTensor) -> str:
    decoded = _safe_batch_decode(tokenizer, sequences, skip_special_tokens=False)[0]
    start_tags = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
    end_tag = "<|im_end|>"
    resp = decoded
    for st in start_tags:
        if st in decoded:
            resp = decoded.split(st)[-1]
            break
    if "<|im_end|>assistant" in resp:
        resp = resp.split("<|im_end|>assistant")[-1]
    if end_tag in resp:
        resp = resp.split(end_tag)[0]
    for cutter in ["\n###", "###\n", "###"]:
        if cutter in resp:
            resp = resp.split(cutter)[0]
            break
    return resp.strip()


def _first_nonempty_line(text: str) -> str:
    if not isinstance(text, str):
        return text
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def _extract_single_choice_letter(text: Optional[str], valid_letters: Sequence[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    pattern = rf"^[^A-Za-z]*([{''.join(valid_letters)}])(?:\b|[^A-Za-z])"
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        m = re.match(pattern, line, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        break
    seen = []
    for ch in text.upper():
        if ch in valid_letters and ch not in seen:
            seen.append(ch)
    if len(seen) == 1:
        return seen[0]
    return None


def _extract_choice_set(text: Optional[str], valid_letters: Sequence[str]) -> List[str]:
    if not isinstance(text, str):
        return []
    letters: List[str] = []
    for ch in text.upper():
        if ch in valid_letters and ch not in letters:
            letters.append(ch)
    return sorted(letters)


def _strip_option_prefix(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    m = OPTION_PREFIX_RE.match(text)
    if m:
        return text[m.end() :].strip()
    return text.strip()


def _load_mm_extras_if_missing(model, ckpt_dir: str):
    if safe_load_file is None:
        return
    index_path = Path(ckpt_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        return
    data = json.loads(index_path.read_text())
    weight_map: Dict[str, str] = data.get("weight_map", {})
    target_prefixes = ("model.multiscale_token_mlp", "model.text_to_vision_proj", "model.scale_pos_residuals")
    target_keys = [k for k in weight_map.keys() if k.startswith(target_prefixes)]
    if not target_keys:
        return

    shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    sub_state: Dict[str, torch.Tensor] = {}
    for k in target_keys:
        shard_file = weight_map[k]
        shard_path = Path(ckpt_dir) / shard_file
        if shard_file not in shard_cache:
            shard_cache[shard_file] = safe_load_file(str(shard_path))
        shard = shard_cache[shard_file]
        if k in shard:
            sub_state[k] = shard[k]
    if sub_state:
        model.load_state_dict(sub_state, strict=False)


def prepare_multiscale_inputs(
    image_paths: Sequence[str],
    image_processor,
    image_root: str,
    mask_root: str,
    target_sizes: Sequence[int],
    tile_size: int,
    patch_size: int,
    allow_missing_mask: bool = False,
) -> Tuple[List[Dict[int, Dict]], List[Optional[Dict[int, Dict]]], List[Tuple[int, int]]]:
    multiscale_pixels: List[Dict[int, Dict]] = []
    multiscale_masks: List[Optional[Dict[int, Dict]]] = []
    all_masks_present = True
    image_sizes: List[Tuple[int, int]] = []
    for rel in image_paths:
        abs_path = os.path.join(os.path.abspath(image_root), rel)
        img = Image.open(abs_path).convert("RGB")
        image_sizes.append(img.size)
        tiles = split_image_to_multiscale_tiles(
            img,
            image_processor,
            target_sizes=target_sizes,
            tile_size=tile_size,
        )
        mask_path = os.path.join(mask_root, f"{_sanitize_cache_name(rel)}.pt") if mask_root else None
        masks = None
        if mask_path and os.path.isfile(mask_path):
            try:
                mask_entry = torch.load(mask_path, map_location="cpu")
                masks = load_multiscale_tile_masks(
                    mask_entry,
                    scales=target_sizes,
                    tile_size=tile_size,
                    patch_size=patch_size,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load mask for {rel}: {exc}") from exc
            if not masks:
                if not allow_missing_mask:
                    raise FileNotFoundError(f"Mask empty or invalid for {rel}: {mask_path}")
                all_masks_present = False
                masks = None
        elif not allow_missing_mask:
            raise FileNotFoundError(f"Mask not found for {rel}: {mask_path}")
        else:
            all_masks_present = False
            masks = None
        multiscale_pixels.append(tiles)
        multiscale_masks.append(masks)
    if not all_masks_present:
        multiscale_masks = None
    return multiscale_pixels, multiscale_masks, image_sizes


def build_prompt_and_inputs(tokenizer, conv, question: str, image_tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    conv.messages = []
    cleaned_question = _strip_leading_image_tokens(question) if image_tokens else question
    if image_tokens and "<image>" not in cleaned_question:
        cleaned_question = "<image>\n" + cleaned_question
    human_message = (cleaned_question, image_tokens) if image_tokens else cleaned_question
    conv.append_message(conv.roles[0], human_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def _format_question(sample: Dict[str, Any], valid_letters: Sequence[str]) -> Tuple[str, List[str]]:
    text = str(sample.get("Text", "") or "").strip()
    options = sample.get("Answer choices") or sample.get("answer_choices") or sample.get("options") or []
    options = list(options) if isinstance(options, (list, tuple)) else []
    letters = list(valid_letters[: len(options)]) if options else list(valid_letters)
    lines = [text] if text else ["Answer the question based on the image."]
    if options:
        lines.append("Options:")
        for letter, opt in zip(letters, options):
            lines.append(f"{letter}. {_strip_option_prefix(str(opt))}")
        lines.append("Please reply with the letter of the correct option (e.g., A/B/C/D/E).")
    else:
        lines.append("Please reply concisely.")
    return "\n".join(lines), letters


def _extract_from_conversations(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Some inputs are already converted to `conversations` format (like XLRS).
    In that case we reuse the authored question and answer instead of reconstructing
    from Text/Answer choices.
    """
    q_text: Optional[str] = None
    gt: Optional[str] = None
    for msg in sample.get("conversations", []) or []:
        role = str(msg.get("from") or "").lower()
        if role == "human" and q_text is None:
            q_text = msg.get("value", "")
        elif role in {"gpt", "assistant"} and gt is None:
            gt = str(msg.get("value", "")).strip()
    letters_in_question = _extract_choice_set(q_text, CHOICE_LETTERS)
    letters_used = letters_in_question if letters_in_question else list(CHOICE_LETTERS)
    return q_text, gt, letters_used


def _determine_correct(pred_text: str, gt_text: str, valid_letters: Sequence[str]) -> Tuple[bool, List[str], List[str]]:
    gt_choices = _extract_choice_set(gt_text, valid_letters)
    pred_choices = _extract_choice_set(pred_text, valid_letters)
    if gt_choices:
        return sorted(pred_choices) == sorted(gt_choices), pred_choices, gt_choices
    return pred_text.strip().lower() == str(gt_text or "").strip().lower(), pred_choices, gt_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MME-RealWorld-Lite (Remote Sensing) with multiscale pruning.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    parser.add_argument("--model_name", default="llava_qwen", help="Model name hint for loader.")
    parser.add_argument(
        "--data_json",
        default="/path/to/mme_realworld_lite.json",
        help="Path to the MME-RealWorld-Lite annotation JSON file.",
    )
    parser.add_argument(
        "--image_root",
        default="/path/to/mme_images_root",
        help="Root folder containing images referenced by data_json.",
    )
    parser.add_argument(
        "--mask_root",
        default="/path/to/multiscale_tile_mask_dir",
        help="Directory of multiscale tile masks (.pt) generated by build_multiscale_tile_masks.py.",
    )
    parser.add_argument("--output_json", default="mme_remote_sensing_results.jsonl", help="Output JSONL path.")
    parser.add_argument("--subtask", default="Remote Sensing", help="Only evaluate this Subtask (case-insensitive).")
    parser.add_argument("--devices", default=None, help="Comma-separated devices for parallel eval, e.g., cuda:0,cuda:1.")
    parser.add_argument("--device", default="cuda:0", help="Device when --devices is not set.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples (0 for all).")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume; overwrite output file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("--multiscale_topk", default="80,320,600,2000", help="Comma-separated top-k per scale.")
    parser.add_argument("--multiscale_target_sizes", default="672,1344,2688,4032", help="Comma-separated square sizes.")
    parser.add_argument("--tile_size", type=int, default=336, help="Tile size used in preprocessing.")
    parser.add_argument("--patch_size", type=int, default=14, help="Vision patch size inside a tile.")
    parser.add_argument("--allow_missing_mask", action="store_true", help="Fallback to standard pipeline when mask is missing.")
    parser.add_argument("--prompt_version", default="qwen_1_5", help="Conversation template key.")
    return parser.parse_args()


def _prepare_devices(args: argparse.Namespace) -> List[str]:
    if args.devices:
        devices_list = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    else:
        if "," in str(args.device):
            devices_list = [d.strip() for d in str(args.device).split(",") if d.strip()]
        else:
            devices_list = [args.device]
    if not devices_list:
        devices_list = ["cuda:0"]
    return devices_list


def run_worker(
    rank: int,
    indices: List[int],
    data: List[Dict[str, Any]],
    args: argparse.Namespace,
    out_path: Path,
    lock: Any,
) -> None:
    device = args.devices_list[rank]

    gen_kwargs = dict(
        do_sample=bool(args.temperature and args.temperature > 1e-8) or (args.top_p is not None and args.top_p < 1.0),
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=1,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
    )
    gen_kwargs["min_new_tokens"] = max(1, gen_kwargs.get("min_new_tokens", 1))
    if not gen_kwargs["do_sample"]:
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)

    tokenizer, model, image_processor, _ = load_pretrained_model(args.ckpt, None, args.model_name, device_map=device)
    try:
        _load_mm_extras_if_missing(model, args.ckpt)
    except Exception as exc:
        print(f"[Warn] failed to load multiscale extras: {exc}")
    try:
        model.generation_config.repetition_penalty = 1.0
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model.config._log_vision_tokens = False
    except Exception:
        pass
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        bos_id = getattr(model.generation_config, "bos_token_id", None)
        if isinstance(bos_id, (list, tuple)) and bos_id:
            bos_id = bos_id[0]
        if bos_id is None:
            bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        model.generation_config.bos_token_id = int(bos_id)
        if not gen_kwargs["do_sample"]:
            model.generation_config.top_k = 0
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = 1.0
            if hasattr(model.generation_config, "top_p"):
                model.generation_config.top_p = 1.0
    if getattr(model, "config", None) is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if getattr(model.config, "bos_token_id", None) is None:
            model.config.bos_token_id = getattr(model.generation_config, "bos_token_id", tokenizer.bos_token_id)

    conv = conv_templates[args.prompt_version].copy()
    target_sizes = _parse_int_list(args.multiscale_target_sizes)
    topk = _parse_int_list(args.multiscale_topk)
    model_dtype = next(model.parameters()).dtype

    for idx in indices:
        sample = data[idx]
        try:
            subtask = str(sample.get("Subtask", "") or "").strip()
            if args.subtask and subtask and subtask.lower() != args.subtask.lower():
                continue
            image_field = sample.get("Image") or sample.get("image")
            rel_list = [image_field] if isinstance(image_field, str) else list(image_field or [])
            if not rel_list:
                raise ValueError("Sample missing image path.")
            abs_list = [os.path.join(args.image_root, rel) for rel in rel_list]
            for p in abs_list:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Image missing: {p}")

            conv_q, conv_gt, conv_letters = _extract_from_conversations(sample)

            options = sample.get("Answer choices") or sample.get("answer_choices") or sample.get("options") or []
            options = list(options) if isinstance(options, (list, tuple)) else []
            valid_letters = CHOICE_LETTERS[: max(1, min(len(options), len(CHOICE_LETTERS)))] if options else (
                conv_letters if conv_letters else CHOICE_LETTERS
            )

            if conv_q is not None:
                q_text = conv_q
                letters_used = conv_letters if conv_letters else list(valid_letters)
            else:
                q_text, letters_used = _format_question(sample, valid_letters)

            gt = str(sample.get("Ground truth") or sample.get("answer") or "").strip()
            if not gt and conv_gt:
                gt = conv_gt
            task = str(sample.get("Task") or "unknown").strip() or "unknown"
            category = str(sample.get("Category") or "unknown").strip() or "unknown"

            pil_list = [Image.open(p).convert("RGB") for p in abs_list]
            image_sizes = [img.size for img in pil_list]
            images_payload = process_images(pil_list, image_processor, model.config)
            if isinstance(images_payload, torch.Tensor):
                images_payload = images_payload.to(model.device, dtype=model_dtype)
            elif isinstance(images_payload, list):
                try:
                    if len(images_payload) == 1:
                        images_payload = images_payload[0].unsqueeze(0)
                    elif all(x.shape == images_payload[0].shape for x in images_payload):
                        images_payload = torch.stack(images_payload)
                    else:
                        images_payload = image_processor(pil_list, return_tensors="pt")["pixel_values"]
                    images_payload = images_payload.to(model.device, dtype=model_dtype)
                except Exception:
                    images_payload = [im.to(model.device, dtype=model_dtype) for im in images_payload]
            else:
                images_payload = image_processor(pil_list, return_tensors="pt")["pixel_values"].to(
                    model.device, dtype=model_dtype
                )

            ms_pixels, ms_masks, image_sizes = prepare_multiscale_inputs(
                rel_list,
                image_processor,
                image_root=args.image_root,
                mask_root=args.mask_root,
                target_sizes=target_sizes,
                tile_size=args.tile_size,
                patch_size=args.patch_size,
                allow_missing_mask=args.allow_missing_mask,
            )
            if ms_masks is None:
                raise RuntimeError(
                    f"multiscale_masks is None (mask files missing or invalid under {args.mask_root}); "
                    "cannot run multiscale pruning."
                )

            input_ids, attention_mask = build_prompt_and_inputs(tokenizer, conv, q_text, rel_list)
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
            image_token_id = getattr(model.config, "image_token_index", IMAGE_TOKEN_INDEX)
            if vocab_size is not None:
                oov_mask = (input_ids >= vocab_size) | ((input_ids < 0) & (input_ids != image_token_id))
                if oov_mask.any():
                    input_ids = input_ids.masked_fill(oov_mask, tokenizer.eos_token_id)
            if (input_ids == image_token_id).sum() == 0 and images_payload is not None:
                image_tok = torch.tensor([[image_token_id]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([image_tok, input_ids], dim=1)
                pad = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([pad, attention_mask], dim=1)

            with torch.inference_mode():
                stop_strings = [conv.sep]
                if getattr(conv, "sep2", None):
                    stop_strings.append(conv.sep2)
                stop_strings.extend(["\n###", "###", "<|im_start|>", "<|im_end|>"])
                stopping = KeywordsStoppingCriteria(stop_strings, tokenizer, input_ids)
                output = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=images_payload,
                    image_sizes=image_sizes,
                    modalities=["image"] * len(rel_list),
                    multiscale_pixels=ms_pixels,
                    multiscale_masks=ms_masks,
                    multiscale_topk=topk,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    stopping_criteria=[stopping],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                )

            raw = decode_chatml_response(tokenizer, output.sequences)
            raw_gen = _strip_leading_image_tokens(raw)
            clean_output = _first_nonempty_line(raw_gen)
            pred_choice = _extract_single_choice_letter(clean_output, letters_used)
            pred_choices = _extract_choice_set(clean_output, letters_used)
            used_logit_fallback = False
            if (
                (clean_output is None or clean_output.strip() in {"", "!"} or len(clean_output.strip()) == 1)
                and getattr(output, "scores", None)
            ):
                try:
                    letter_ids = {}
                    for ch in letters_used:
                        ids = tokenizer.encode(ch, add_special_tokens=False)
                        if isinstance(ids, list) and len(ids) == 1:
                            letter_ids[ch] = ids[0]
                    if letter_ids:
                        step0 = output.scores[0][0]
                        best_letter = max(letter_ids.items(), key=lambda kv: step0[kv[1]].item())[0]
                        clean_output = best_letter
                        pred_choice = best_letter
                        pred_choices = [best_letter]
                        used_logit_fallback = True
                except Exception:
                    pass
            if pred_choices:
                clean_output = "".join(sorted(pred_choices))
            elif pred_choice:
                clean_output = pred_choice
            elif clean_output is None:
                clean_output = _first_nonempty_line(raw)

            correct, pred_set, gt_set = _determine_correct(clean_output, gt, letters_used)

            result = {
                "id": sample.get("Question_id") or sample.get("id") or idx,
                "image": rel_list,
                "question": q_text,
                "ground_truth": gt,
                "model_output": clean_output,
                "raw_output": raw,
                "raw_tail": raw_gen,
                "image_sizes": image_sizes,
                "logit_fallback": used_logit_fallback if used_logit_fallback else False,
                "task": task,
                "category": category,
                "subtask": subtask or args.subtask,
                "correct": bool(correct),
                "pred_choices": pred_set,
                "gt_choices": gt_set,
            }

            line = json.dumps(result, ensure_ascii=False) + "\n"
            _append_line(lock, out_path, line)
        except Exception as exc:
            err_obj = {
                "id": sample.get("Question_id") or sample.get("id") or idx,
                "image": sample.get("Image") or sample.get("image"),
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
            _append_line(lock, out_path, json.dumps(err_obj, ensure_ascii=False) + "\n")


def _compute_stats(results_path: Path, subtask_filter: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    task_stats = {}
    category_stats = {}
    overall = {"correct": 0, "total": 0}

    def _update(stats: Dict[str, Dict[str, int]], key: str, correct: bool):
        entry = stats.setdefault(key, {"correct": 0, "total": 0})
        entry["total"] += 1
        if correct:
            entry["correct"] += 1

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "correct" not in obj:
                continue
            if subtask_filter:
                st = str(obj.get("subtask", "") or "").strip().lower()
                if st and st != subtask_filter.lower():
                    continue
            correct = bool(obj.get("correct"))
            task = str(obj.get("task") or "unknown").strip() or "unknown"
            category = str(obj.get("category") or "unknown").strip() or "unknown"
            _update(task_stats, task, correct)
            _update(category_stats, category, correct)
            overall["total"] += 1
            if correct:
                overall["correct"] += 1

    return {"task": task_stats, "category": category_stats, "overall": overall}


def _format_stat_lines(stats: Dict[str, Dict[str, int]]) -> List[str]:
    lines = []
    for key, val in sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        total = val.get("total", 0)
        correct = val.get("correct", 0)
        acc = correct / total if total else 0.0
        lines.append(f"{key}: {correct}/{total} ({acc:.2%})")
    return lines


def main() -> None:
    global args
    args = parse_args()
    args.devices_list = _prepare_devices(args)

    raw_data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    filtered = []
    for idx, sample in enumerate(raw_data):
        st = str(sample.get("Subtask", "") or "").strip().lower()
        if args.subtask and st and st != args.subtask.lower():
            continue
        filtered.append(sample)
    if not filtered:
        raise ValueError(f"No samples matched Subtask '{args.subtask}' in {args.data_json}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out_path.exists():
        out_path.unlink()

    resume_enabled = not args.no_resume
    completed_ids = _load_completed_ids(out_path) if resume_enabled else set()
    indices_all: List[int] = []
    for i, sample in enumerate(filtered):
        sid = sample.get("Question_id") or sample.get("id") or i
        if resume_enabled and sid in completed_ids:
            continue
        indices_all.append(i)
    if args.max_samples and args.max_samples > 0:
        indices_all = indices_all[: args.max_samples]

    if len(args.devices_list) == 1:
        lock = None
        with tqdm(total=len(indices_all), desc="Eval", unit="item") as pbar:
            run_worker(0, indices_all, filtered, args, out_path, lock)
            pbar.update(len(indices_all))
        print(f"Done. Wrote {len(indices_all)} results to {out_path}")
    else:
        world_size = len(args.devices_list)
        shards: List[List[int]] = [[] for _ in range(world_size)]
        for i, idx in enumerate(indices_all):
            shards[i % world_size].append(idx)

        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        lock = manager.Lock()
        procs = []
        for rank in range(world_size):
            p = mp.Process(
                target=run_worker,
                args=(rank, shards[rank], filtered, args, out_path, lock),
                daemon=True,
            )
            p.start()
            procs.append(p)

        with tqdm(total=len(indices_all), desc="Eval", unit="item") as pbar:
            while any(p.is_alive() for p in procs):
                done = 0
                if out_path.exists():
                    try:
                        with out_path.open("r", encoding="utf-8") as f:
                            done = sum(1 for _ in f)
                    except Exception:
                        pass
                pbar.n = done
                pbar.refresh()
                time.sleep(0.2)
            try:
                with out_path.open("r", encoding="utf-8") as f:
                    pbar.n = sum(1 for _ in f)
                    pbar.refresh()
            except Exception:
                pass

        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode and p.exitcode != 0]
        if failed:
            print(f"[WARN] {len(failed)} worker(s) exited with errors: {failed}. Partial results may be written.")
        print(f"Done. Wrote up to {len(indices_all)} results to {out_path}")

    stats = _compute_stats(out_path, args.subtask)
    overall = stats.get("overall", {"correct": 0, "total": 0})
    overall_acc = (overall.get("correct", 0) / overall.get("total", 1)) if overall.get("total", 0) else 0.0
    print("\nOverall:", f"{overall.get('correct', 0)}/{overall.get('total', 0)} ({overall_acc:.2%})")
    print("\nPer Task:")
    for line in _format_stat_lines(stats.get("task", {})):
        print("  ", line)
    print("\nPer Category:")
    for line in _format_stat_lines(stats.get("category", {})):
        print("  ", line)


if __name__ == "__main__":
    main()
