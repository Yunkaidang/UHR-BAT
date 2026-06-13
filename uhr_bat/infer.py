#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-image inference entry for UHR-BAT / LongVA.

Example:
  conda activate longva
  cd /path/to/UHR-BAT
  python -m uhr_bat.infer \
    --image /path/to/image.png \
    --question "What objects are in this remote-sensing image?"

For XHRBench images under data/XHRBench you can pass a relative path:
  python -m uhr_bat.infer \
    --image images/000000_1_1亿__14-2012-0415-6905-LA93-0M50-E080.jp2.png \
    --image-root data/XHRBench \
    --question "Describe this satellite image."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:  # pragma: no cover - optional dependency
    safe_load_file = None


Image.MAX_IMAGE_PIXELS = None

THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent
CODE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_HF_CKPT = "RL-MIND/UHR-BAT"
_LOCAL_CKPT_CANDIDATES = [
    CODE_ROOT / "checkpoints" / "UHR-BAT",
    PROJECT_ROOT / "UHR-BAT",
]
_LOCAL_IMAGE_ROOT_CANDIDATES = [
    CODE_ROOT / "data" / "XHRBench",
    CODE_ROOT / "data" / "UHR-BAT-SFT-10K" / "train" / "images",
    PROJECT_ROOT / "UHR-BAT-Data" / "jpg_images",
]
DEFAULT_CKPT = os.environ.get("UHR_BAT_CKPT") or next(
    (str(path) for path in _LOCAL_CKPT_CANDIDATES if path.exists()),
    DEFAULT_HF_CKPT,
)
DEFAULT_IMAGE_ROOT = os.environ.get("UHR_BAT_IMAGE_ROOT") or next(
    (str(path) for path in _LOCAL_IMAGE_ROOT_CANDIDATES if path.exists()),
    str(CODE_ROOT / "data" / "XHRBench"),
)
DEFAULT_KMEANS_LONGVA = CODE_ROOT / "with_k-means" / "longva"
DEFAULT_SAM_LONGVA = CODE_ROOT / "with-SAM" / "longva"

_LONGVA_IMPORTED = False


_SPECIAL_TOKEN_CLEANER = re.compile(
    r"<\|[^>]+?\|>(?: ?assistant)?|<image>|</?s>|<pad>|<eos>|<unk>",
    flags=re.IGNORECASE,
)
_LEADING_IMAGE_TOKEN_PATTERN = re.compile(r"^((?:\s*<image>\s*)+)", re.IGNORECASE)


def _import_longva(longva_root: Path):
    """
    Import the branch-local longva package after putting it first on sys.path.
    """
    global _LONGVA_IMPORTED

    longva_root = Path(longva_root).expanduser().resolve()
    if not longva_root.exists():
        raise FileNotFoundError(f"LongVA path not found: {longva_root}")

    root_str = str(longva_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    if not _LONGVA_IMPORTED:
        try:
            import longva.utils as _lv_utils

            _lv_utils.rank0_print = print
        except Exception:
            pass
        _LONGVA_IMPORTED = True

    from longva.conversation import conv_templates
    from longva.constants import IMAGE_TOKEN_INDEX
    from longva.mm_utils import (
        KeywordsStoppingCriteria,
        load_multiscale_tile_masks,
        process_images,
        split_image_to_multiscale_tiles,
        tokenizer_image_token,
    )
    from longva.model.builder import load_pretrained_model

    return SimpleNamespace(
        conv_templates=conv_templates,
        IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
        KeywordsStoppingCriteria=KeywordsStoppingCriteria,
        load_multiscale_tile_masks=load_multiscale_tile_masks,
        process_images=process_images,
        split_image_to_multiscale_tiles=split_image_to_multiscale_tiles,
        tokenizer_image_token=tokenizer_image_token,
        load_pretrained_model=load_pretrained_model,
    )


def _sanitize_cache_name(image_rel_path: str) -> str:
    normalized = os.path.normpath(str(image_rel_path).strip())
    return normalized.replace("\\", "__").replace("/", "__")


def _strip_leading_image_tokens(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    match = _LEADING_IMAGE_TOKEN_PATTERN.match(text)
    if not match:
        return text
    remainder = text[match.end() :].lstrip("\r\n")
    removed = match.group(1)
    if "\n" in removed and remainder and not remainder.startswith("\n"):
        remainder = "\n" + remainder
    return remainder


def _parse_int_list(value: Sequence[int] | str) -> List[int]:
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _clean_generated_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = _SPECIAL_TOKEN_CLEANER.sub(" ", str(text))
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def _safe_batch_decode(tokenizer, sequences, image_token_index: int, **decode_kwargs) -> List[str]:
    try:
        seq_tensor = torch.as_tensor(sequences, dtype=torch.long)
    except Exception:
        return tokenizer.batch_decode(sequences, **decode_kwargs)

    replace_id = tokenizer.pad_token_id
    if replace_id is None:
        replace_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    seq_tensor = seq_tensor.clone()
    seq_tensor = seq_tensor.masked_fill(seq_tensor == image_token_index, replace_id)
    seq_tensor = seq_tensor.masked_fill(seq_tensor < 0, replace_id)
    if vocab_size:
        seq_tensor = seq_tensor.masked_fill(seq_tensor >= vocab_size, replace_id)
    return tokenizer.batch_decode(seq_tensor, **decode_kwargs)


def _decode_chatml_response(tokenizer, sequences, image_token_index: int) -> str:
    decoded = _safe_batch_decode(tokenizer, sequences, image_token_index, skip_special_tokens=False)[0]
    response = decoded
    for start_tag in ("<|im_start|>assistant\n", "<|im_start|>assistant"):
        if start_tag in decoded:
            response = decoded.split(start_tag)[-1]
            break
    if "<|im_end|>assistant" in response:
        response = response.split("<|im_end|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    for cutter in ("\n###", "###\n", "###"):
        if cutter in response:
            response = response.split(cutter)[0]
            break
    return response.strip()


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        line = line.strip()
        if line:
            return line
    return str(text).strip()


def _resolve_image_path(image: str | Path, image_root: str | Path) -> Tuple[Path, str]:
    """
    Return absolute image path and a stable relative name used for mask-cache lookup.
    """
    image_root = Path(image_root).expanduser().resolve()
    image_path = Path(image).expanduser()
    if not image_path.is_absolute():
        image_path = image_root / image_path
    image_path = image_path.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        rel_name = image_path.relative_to(image_root).as_posix()
    except ValueError:
        rel_name = image_path.name
    return image_path, rel_name


def _move_images_payload(images_payload, device, dtype):
    if isinstance(images_payload, torch.Tensor):
        return images_payload.to(device=device, dtype=dtype)
    if isinstance(images_payload, list):
        try:
            if len(images_payload) == 1:
                images_payload = images_payload[0].unsqueeze(0)
            elif all(x.shape == images_payload[0].shape for x in images_payload):
                images_payload = torch.stack(images_payload)
            else:
                return [im.to(device=device, dtype=dtype) for im in images_payload]
            return images_payload.to(device=device, dtype=dtype)
        except Exception:
            return [im.to(device=device, dtype=dtype) for im in images_payload]
    return images_payload


def _load_mm_extras_if_missing(model, ckpt_dir: str | Path) -> None:
    """
    Some merged HF checkpoints skip UHR-BAT extras during from_pretrained.
    This reloads the multiscale modules explicitly when a safetensors index is present.
    """
    if safe_load_file is None:
        return
    ckpt_dir = Path(ckpt_dir)
    index_path = ckpt_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return

    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = data.get("weight_map", {})
    target_prefixes = ("model.multiscale_token_mlp", "model.text_to_vision_proj", "model.scale_pos_residuals")
    target_keys = [k for k in weight_map if k.startswith(target_prefixes)]
    if not target_keys:
        return

    shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    sub_state: Dict[str, torch.Tensor] = {}
    for key in target_keys:
        shard_file = weight_map[key]
        if shard_file not in shard_cache:
            shard_cache[shard_file] = safe_load_file(str(ckpt_dir / shard_file))
        if key in shard_cache[shard_file]:
            sub_state[key] = shard_cache[shard_file][key]

    if sub_state:
        model.load_state_dict(sub_state, strict=False)


def _load_model(
    lv,
    ckpt: str | Path,
    model_name: str,
    device: str,
    load_8bit: bool = False,
    load_4bit: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    vision_tower: Optional[str | Path] = None,
):
    overwrite_config = None
    if vision_tower:
        vt = str(Path(vision_tower).expanduser().resolve())
        overwrite_config = {
            "mm_vision_tower": vt,
            "vision_tower": vt,
        }

    tokenizer, model, image_processor, context_len = lv.load_pretrained_model(
        str(ckpt),
        None,
        model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map=device,
        attn_implementation=attn_implementation,
        overwrite_config=overwrite_config,
    )
    _load_mm_extras_if_missing(model, ckpt)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        bos_id = getattr(model.generation_config, "bos_token_id", None)
        if isinstance(bos_id, (list, tuple)) and bos_id:
            bos_id = bos_id[0]
        if bos_id is None:
            bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        if bos_id is not None:
            model.generation_config.bos_token_id = int(bos_id)
    if getattr(model, "config", None) is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if getattr(model.config, "bos_token_id", None) is None:
            model.config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        try:
            model.config._log_vision_tokens = False
        except Exception:
            pass
    return tokenizer, model, image_processor, context_len


def _set_kmeans_config(model, num_clusters: Optional[int], max_iters: Optional[int]) -> None:
    def set_cfg(target, name, value):
        if target is None or value is None:
            return
        cfg = getattr(target, "config", None)
        if cfg is not None:
            try:
                setattr(cfg, name, value)
            except Exception:
                pass

    inner = getattr(model, "get_model", lambda: None)()
    set_cfg(model, "kmeans_num_clusters", num_clusters)
    set_cfg(inner, "kmeans_num_clusters", num_clusters)
    set_cfg(model, "kmeans_max_iters", max_iters)
    set_cfg(inner, "kmeans_max_iters", max_iters)


def _build_prompt_and_inputs(lv, tokenizer, conv, question: str, image_count: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
    conv.messages = []
    cleaned_question = _strip_leading_image_tokens(question)
    if "<image>" not in cleaned_question:
        cleaned_question = "<image>\n" + cleaned_question

    image_tokens = ["image"] * image_count
    human_message = (cleaned_question, image_tokens)
    conv.append_message(conv.roles[0], human_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = lv.tokenizer_image_token(
        prompt,
        tokenizer,
        lv.IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask, prompt


def _prepare_single_image_inputs(
    lv,
    image_path: Path,
    rel_name: str,
    image_processor,
    model,
    target_sizes: Sequence[int],
    tile_size: int,
    patch_size: int,
    mask_root: Optional[str | Path],
    mask_file: Optional[str | Path],
    allow_missing_mask: bool,
) -> Tuple[Any, List[Dict[int, Dict]], List[Dict[int, Dict]], List[Tuple[int, int]], Optional[Path]]:
    pil_image = Image.open(image_path).convert("RGB")
    image_sizes = [pil_image.size]
    model_dtype = next(model.parameters()).dtype

    images_payload = lv.process_images([pil_image], image_processor, model.config)
    if not isinstance(images_payload, (torch.Tensor, list)):
        images_payload = image_processor([pil_image], return_tensors="pt")["pixel_values"]
    images_payload = _move_images_payload(images_payload, model.device, model_dtype)

    multiscale_pixels = [
        lv.split_image_to_multiscale_tiles(
            pil_image,
            image_processor,
            target_sizes=target_sizes,
            tile_size=tile_size,
        )
    ]

    resolved_mask_path = None
    if mask_file:
        candidate = Path(mask_file).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved_mask_path = candidate.resolve()
    elif mask_root:
        resolved_mask_path = Path(mask_root).expanduser().resolve() / f"{_sanitize_cache_name(rel_name)}.pt"

    masks: Dict[int, Dict] = {}
    if resolved_mask_path and resolved_mask_path.is_file():
        mask_entry = torch.load(resolved_mask_path, map_location="cpu")
        masks = lv.load_multiscale_tile_masks(
            mask_entry,
            scales=target_sizes,
            tile_size=tile_size,
            patch_size=patch_size,
        )
        if not masks and not allow_missing_mask:
            raise RuntimeError(f"Mask file is empty or invalid: {resolved_mask_path}")
    elif resolved_mask_path and not allow_missing_mask:
        raise FileNotFoundError(f"Mask not found: {resolved_mask_path}")

    return images_payload, multiscale_pixels, [masks], image_sizes, resolved_mask_path


def run_single_image_inference(
    image: str | Path,
    question: str,
    ckpt: str | Path = DEFAULT_CKPT,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    longva_root: Optional[str | Path] = None,
    branch: str = "kmeans",
    model_name: str = "llava_qwen",
    device: str = "cuda:0",
    prompt_version: str = "qwen_1_5",
    mask_root: Optional[str | Path] = None,
    mask_file: Optional[str | Path] = None,
    allow_missing_mask: bool = True,
    multiscale_topk: Sequence[int] | str = "80,320,600,2000",
    multiscale_target_sizes: Sequence[int] | str = "672,1344,2688,4032",
    tile_size: int = 336,
    patch_size: int = 14,
    kmeans_num_clusters: Optional[int] = None,
    kmeans_max_iters: int = 10,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_new_tokens: int = 512,
    load_8bit: bool = False,
    load_4bit: bool = False,
    attn_implementation: Optional[str] = "flash_attention_2",
    vision_tower: Optional[str | Path] = None,
    include_raw_full: bool = False,
) -> Dict[str, Any]:
    """
    Run UHR-BAT / LongVA inference on one image and return a JSON-serializable result.
    """
    if longva_root is None:
        longva_root = DEFAULT_SAM_LONGVA if branch.lower() == "sam" else DEFAULT_KMEANS_LONGVA

    lv = _import_longva(Path(longva_root))
    image_path, rel_name = _resolve_image_path(image, image_root)
    target_sizes = _parse_int_list(multiscale_target_sizes)
    topk = _parse_int_list(multiscale_topk)

    tokenizer, model, image_processor, context_len = _load_model(
        lv,
        ckpt=ckpt,
        model_name=model_name,
        device=device,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        attn_implementation=attn_implementation,
        vision_tower=vision_tower,
    )
    if branch.lower() in {"kmeans", "with_k-means", "with-kmeans"}:
        _set_kmeans_config(model, kmeans_num_clusters, kmeans_max_iters)

    conv = lv.conv_templates[prompt_version].copy()
    input_ids, attention_mask, prompt = _build_prompt_and_inputs(lv, tokenizer, conv, question, image_count=1)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    image_token_id = getattr(model.config, "image_token_index", lv.IMAGE_TOKEN_INDEX)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    if vocab_size is not None:
        oov_mask = (input_ids >= vocab_size) | ((input_ids < 0) & (input_ids != image_token_id))
        if oov_mask.any():
            input_ids = input_ids.masked_fill(oov_mask, tokenizer.eos_token_id)
    if (input_ids == image_token_id).sum() == 0:
        image_tok = torch.tensor([[image_token_id]], device=input_ids.device, dtype=input_ids.dtype)
        input_ids = torch.cat([image_tok, input_ids], dim=1)
        attention_mask = torch.cat(
            [torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask],
            dim=1,
        )

    images_payload, ms_pixels, ms_masks, image_sizes, resolved_mask_path = _prepare_single_image_inputs(
        lv,
        image_path=image_path,
        rel_name=rel_name,
        image_processor=image_processor,
        model=model,
        target_sizes=target_sizes,
        tile_size=tile_size,
        patch_size=patch_size,
        mask_root=mask_root,
        mask_file=mask_file,
        allow_missing_mask=allow_missing_mask,
    )

    do_sample = bool(temperature and temperature > 1e-8) or (top_p is not None and top_p < 1.0)
    gen_kwargs = {
        "do_sample": do_sample,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    elif getattr(model, "generation_config", None) is not None:
        model.generation_config.top_k = 0
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = 1.0
        if hasattr(model.generation_config, "top_p"):
            model.generation_config.top_p = 1.0

    stop_strings = [conv.sep]
    if getattr(conv, "sep2", None):
        stop_strings.append(conv.sep2)
    stop_strings.extend(["\n###", "###", "<|im_start|>", "<|im_end|>"])
    stopping = lv.KeywordsStoppingCriteria(stop_strings, tokenizer, input_ids)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=images_payload,
            image_sizes=image_sizes,
            modalities=["image"],
            multiscale_pixels=ms_pixels,
            multiscale_masks=ms_masks,
            multiscale_topk=topk,
            multiscale_target_sizes=target_sizes,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=[stopping],
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    latency_s = time.perf_counter() - start

    sequences = output.sequences
    gen_steps = len(output.scores) if getattr(output, "scores", None) is not None else 0
    prompt_len = sequences.shape[1] - gen_steps if gen_steps > 0 else input_ids.shape[1]
    gen_tokens = sequences[:, prompt_len:]

    raw_tail = _safe_batch_decode(tokenizer, gen_tokens, lv.IMAGE_TOKEN_INDEX, skip_special_tokens=False)[0]
    decoded_tail = _decode_chatml_response(tokenizer, gen_tokens, lv.IMAGE_TOKEN_INDEX)
    answer = _clean_generated_text(raw_tail) or _clean_generated_text(decoded_tail)
    if not answer:
        answer = _clean_generated_text(_decode_chatml_response(tokenizer, sequences, lv.IMAGE_TOKEN_INDEX))
    answer = _first_nonempty_line(answer)

    result = {
        "image": str(image_path),
        "image_rel": rel_name,
        "question": question,
        "answer": answer,
        "raw_tail": raw_tail,
        "image_sizes": image_sizes,
        "gen_tokens": gen_steps,
        "latency_s": latency_s,
        "checkpoint": str(Path(ckpt).expanduser()),
        "longva_root": str(Path(longva_root).expanduser()),
        "branch": branch,
        "context_len": context_len,
        "multiscale_topk": topk,
        "multiscale_target_sizes": target_sizes,
        "mask_path": str(resolved_mask_path) if resolved_mask_path else None,
        "mask_used": bool(ms_masks and ms_masks[0]),
    }
    if include_raw_full:
        result["raw_full"] = _safe_batch_decode(tokenizer, sequences, lv.IMAGE_TOKEN_INDEX, skip_special_tokens=False)[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UHR-BAT / LongVA inference on one image.")
    parser.add_argument("--image", required=True, help="Image path. Relative paths are resolved under --image-root.")
    parser.add_argument("--question", required=True, help="Question/prompt for the image.")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Checkpoint directory.")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="Root for relative image paths.")
    parser.add_argument(
        "--branch",
        default="kmeans",
        choices=["kmeans", "sam"],
        help="Which branch-local LongVA code to use.",
    )
    parser.add_argument("--longva-root", default=None, help="Override path to a LongVA repo/package root.")
    parser.add_argument("--model-name", default="llava_qwen", help="Model name hint for LongVA loader.")
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0.")
    parser.add_argument("--prompt-version", default="qwen_1_5", help="Conversation template key.")
    parser.add_argument("--mask-root", default=None, help="Optional mask cache directory.")
    parser.add_argument("--mask-file", default=None, help="Optional exact .pt mask file for this image.")
    parser.add_argument(
        "--require-mask",
        action="store_true",
        help="Raise an error when the mask is missing. By default missing masks are allowed.",
    )
    parser.add_argument("--multiscale-topk", default="80,320,600,2000", help="Comma-separated top-k per scale.")
    parser.add_argument("--multiscale-target-sizes", default="672,1344,2688,4032", help="Comma-separated scale sizes.")
    parser.add_argument("--tile-size", type=int, default=336, help="Tile size.")
    parser.add_argument("--patch-size", type=int, default=14, help="Vision patch size.")
    parser.add_argument("--kmeans-num-clusters", type=int, default=None, help="Optional fixed K-Means clusters.")
    parser.add_argument("--kmeans-max-iters", type=int, default=10, help="K-Means iterations.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit.")
    quant.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit.")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation passed to LongVA loader. Use 'none' to disable override.",
    )
    parser.add_argument(
        "--vision-tower",
        default=None,
        help="Optional local vision tower path to override ckpt config (useful in offline mode).",
    )
    parser.add_argument("--include-raw-full", action="store_true", help="Include full decoded sequence in output JSON.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the result JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attn_impl = None if str(args.attn_implementation).lower() in {"none", "null", "false"} else args.attn_implementation
    result = run_single_image_inference(
        image=args.image,
        question=args.question,
        ckpt=args.ckpt,
        image_root=args.image_root,
        longva_root=args.longva_root,
        branch=args.branch,
        model_name=args.model_name,
        device=args.device,
        prompt_version=args.prompt_version,
        mask_root=args.mask_root,
        mask_file=args.mask_file,
        allow_missing_mask=not args.require_mask,
        multiscale_topk=args.multiscale_topk,
        multiscale_target_sizes=args.multiscale_target_sizes,
        tile_size=args.tile_size,
        patch_size=args.patch_size,
        kmeans_num_clusters=args.kmeans_num_clusters,
        kmeans_max_iters=args.kmeans_max_iters,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        attn_implementation=attn_impl,
        vision_tower=args.vision_tower,
        include_raw_full=args.include_raw_full,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
