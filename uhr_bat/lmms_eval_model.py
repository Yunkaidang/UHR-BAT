"""UHR-BAT adapter for EvolvingLMMs-Lab/lmms-eval.

The adapter uses the Hugging Face checkpoint remote code, so it does not require
copying the branch-local LongVA package into lmms-eval. Install this repository
in the same environment as lmms-eval and select it with ``--model uhr_bat``.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import PIL.Image
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")

DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_INDEX = -200

_LEADING_IMAGE_TOKEN_PATTERN = re.compile(r"^((?:\s*<image>\s*)+)", re.IGNORECASE)
_SPECIAL_TOKEN_CLEANER = re.compile(
    r"<\|[^>]+?\|>(?: ?assistant)?|<image>|</?s>|<pad>|<eos>|<unk>",
    flags=re.IGNORECASE,
)


def _parse_int_list(value: Sequence[int] | str | None, default: Sequence[int]) -> List[int]:
    if value is None:
        return [int(x) for x in default]
    if isinstance(value, str):
        cleaned = value.strip().strip("\"'")
        for sep in (";", ":", "|"):
            cleaned = cleaned.replace(sep, ",")
        return [int(x) for x in cleaned.split(",") if x.strip()]
    return [int(x) for x in value]


def _parse_dtype(value: str | torch.dtype):
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower()
    if normalized in {"auto", "none"}:
        return "auto"
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")


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


def _clean_generated_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = _SPECIAL_TOKEN_CLEANER.sub(" ", str(text))
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def _sanitize_cache_name(image_rel_path: str) -> str:
    normalized = os.path.normpath(str(image_rel_path).strip())
    return normalized.replace("\\", "__").replace("/", "__")


@register_model("uhr_bat", "uhr-bat", "uhrbat")
class UHRBAT(lmms):
    """Simple lmms-eval backend for UHR-BAT image understanding."""

    is_simple = True

    def __init__(
        self,
        pretrained: str = "RL-MIND/UHR-BAT",
        device: Optional[str] = "cuda:0",
        device_map: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        multiscale_topk: Union[str, Sequence[int]] = "80:320:600:2000",
        multiscale_target_sizes: Union[str, Sequence[int]] = "672:1344:2688:4032",
        tile_size: int = 336,
        patch_size: int = 14,
        mask_root: Optional[str] = None,
        image_root: Optional[str] = None,
        allow_missing_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning("Ignoring unknown UHR-BAT model_args: %s", sorted(kwargs))

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator

        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = {"": str(self._device)}
        elif device_map == "auto":
            self._device = torch.device(device or "cuda:0")
            self.device_map = "auto"
        else:
            self._device = torch.device(device or f"cuda:{accelerator.local_process_index}")
            self.device_map = {"": str(self._device)}

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)
        self.system_prompt = self._resolve_system_prompt(system_prompt)
        self.default_max_new_tokens = int(max_new_tokens)
        self.default_temperature = float(temperature)
        self.default_top_p = top_p
        self.multiscale_topk = _parse_int_list(multiscale_topk, [80, 320, 600, 2000])
        self.multiscale_target_sizes = _parse_int_list(multiscale_target_sizes, [672, 1344, 2688, 4032])
        self.tile_size = int(tile_size)
        self.patch_size = int(patch_size)
        self.mask_root = Path(mask_root).expanduser().resolve() if mask_root else None
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None
        self.allow_missing_mask = bool(allow_missing_mask)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": _parse_dtype(torch_dtype),
            "device_map": self.device_map,
        }
        if attn_implementation and str(attn_implementation).lower() not in {"none", "null", "false"}:
            load_kwargs["attn_implementation"] = attn_implementation

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._image_processor = AutoImageProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs).eval()
        self._config = self._model.config
        self._uhrbat = importlib.import_module(self.model.__class__.__module__)
        self._image_token_index = int(getattr(self._config, "image_token_index", IMAGE_TOKEN_INDEX))

        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        try:
            self._config._log_vision_tokens = False
        except Exception:
            pass

        assert self.batch_size_per_gpu == 1, "UHR-BAT lmms-eval adapter currently supports batch_size=1."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type. Use DDP/FSDP/DeepSpeed for multi-GPU eval."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
            self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        try:
            return self.model.device
        except Exception:
            return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _flatten(self, values: Iterable[Any]) -> List[Any]:
        flattened: List[Any] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                flattened.extend(value)
            elif value is not None:
                flattened.append(value)
        return flattened

    def _coerce_visuals(self, visuals: Sequence[Any]) -> List[PIL.Image.Image]:
        images: List[PIL.Image.Image] = []
        for visual in visuals:
            if isinstance(visual, PIL.Image.Image):
                image = visual.convert("RGB")
                if getattr(visual, "filename", None):
                    image.filename = visual.filename
                images.append(image)
            elif isinstance(visual, (str, os.PathLike)):
                image = PIL.Image.open(visual).convert("RGB")
                image.filename = str(visual)
                images.append(image)
            else:
                raise TypeError(f"UHR-BAT expects image visuals, got {type(visual).__name__}")
        return images

    def _build_prompt(self, context: str, image_count: int, continuation: Optional[str] = None) -> str:
        user_text = _strip_leading_image_tokens(context).strip()
        if image_count > 0 and DEFAULT_IMAGE_TOKEN not in user_text:
            user_text = " ".join([DEFAULT_IMAGE_TOKEN] * image_count) + "\n" + user_text

        parts: List[str] = []
        if self.system_prompt:
            parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        if continuation is not None:
            parts.append(str(continuation))
            parts.append("<|im_end|>")
        return "".join(parts)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        input_ids = self._uhrbat.tokenizer_image_token(
            prompt,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).unsqueeze(0)
        return input_ids.to(self.device)

    def _safe_decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.detach().clone().cpu()
        replace_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        vocab_size = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)
        ids = ids.masked_fill(ids == self._image_token_index, replace_id)
        ids = ids.masked_fill(ids < 0, replace_id)
        if vocab_size:
            ids = ids.masked_fill(ids >= vocab_size, replace_id)
        return self.tokenizer.decode(ids[0], skip_special_tokens=False)

    def _mask_candidates(self, image: PIL.Image.Image) -> List[Path]:
        if not self.mask_root:
            return []
        candidates: List[Path] = []
        filename = getattr(image, "filename", None)
        if filename:
            path = Path(filename).expanduser()
            rel_names = [path.name]
            try:
                if self.image_root:
                    rel_names.insert(0, path.resolve().relative_to(self.image_root).as_posix())
            except Exception:
                pass
            for rel in rel_names:
                candidates.append(self.mask_root / f"{_sanitize_cache_name(rel)}.pt")
                candidates.append(self.mask_root / f"{Path(rel).name}.pt")
        return list(dict.fromkeys(candidates))

    def _load_mask_for_image(self, image: PIL.Image.Image) -> dict:
        candidates = self._mask_candidates(image)
        for candidate in candidates:
            if candidate.is_file():
                mask_entry = torch.load(candidate, map_location="cpu")
                return self._uhrbat.load_multiscale_tile_masks(
                    mask_entry,
                    scales=self.multiscale_target_sizes,
                    tile_size=self.tile_size,
                    patch_size=self.patch_size,
                )
        if candidates and not self.allow_missing_mask:
            raise FileNotFoundError(f"No mask found. Tried: {[str(x) for x in candidates]}")
        return {}

    def _prepare_multiscale(self, images: Sequence[PIL.Image.Image]) -> Tuple[list, list, list]:
        multiscale_pixels = []
        multiscale_masks = []
        image_sizes = []
        for image in images:
            multiscale_pixels.append(
                self._uhrbat.split_image_to_multiscale_tiles(
                    image,
                    self._image_processor,
                    target_sizes=self.multiscale_target_sizes,
                    tile_size=self.tile_size,
                )
            )
            multiscale_masks.append(self._load_mask_for_image(image))
            image_sizes.append(image.size)
        return multiscale_pixels, multiscale_masks, image_sizes

    def _prepare_gen_kwargs(self, gen_kwargs: dict) -> Tuple[dict, List[str]]:
        gen_kwargs = dict(gen_kwargs or {})
        until = gen_kwargs.pop("until", None)
        stop_strings: List[str] = []
        if isinstance(until, str):
            stop_strings.append(until)
        elif isinstance(until, (list, tuple)):
            stop_strings.extend(str(x) for x in until if x is not None)
        stop_strings.append("<|im_end|>")

        gen_kwargs.pop("image_aspect_ratio", None)
        gen_kwargs.setdefault("max_new_tokens", self.default_max_new_tokens)
        gen_kwargs.setdefault("temperature", self.default_temperature)
        if self.default_top_p is not None:
            gen_kwargs.setdefault("top_p", self.default_top_p)
        gen_kwargs.setdefault("num_beams", 1)

        temperature = float(gen_kwargs.get("temperature") or 0.0)
        top_p = gen_kwargs.get("top_p", None)
        if isinstance(top_p, str) and top_p.lower() in {"none", "null"}:
            top_p = None
            gen_kwargs.pop("top_p", None)
        do_sample = bool(gen_kwargs.get("do_sample", False)) or temperature > 1e-8 or (
            top_p is not None and float(top_p) < 1.0
        )
        gen_kwargs["do_sample"] = do_sample
        if not do_sample:
            gen_kwargs.pop("top_p", None)
            gen_kwargs["temperature"] = 1.0
        return gen_kwargs, list(dict.fromkeys(stop_strings))

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        pbar = tqdm(total=len(chunks), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            task = tasks[0]
            split = splits[0]
            gen_kwargs, stop_strings = self._prepare_gen_kwargs(all_gen_kwargs[0])

            for context, doc_to_visual, doc_id in zip(contexts, doc_to_visuals, doc_ids):
                doc = self.task_dict[task][split][doc_id]
                visuals = self._coerce_visuals(self._flatten([doc_to_visual(doc)]))
                prompt = self._build_prompt(context, len(visuals))
                input_ids = self._tokenize_prompt(prompt)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
                multiscale_pixels, multiscale_masks, image_sizes = self._prepare_multiscale(visuals)
                multimodal_kwargs = {}
                if visuals:
                    multimodal_kwargs = {
                        "image_sizes": image_sizes,
                        "modalities": ["image"],
                        "multiscale_pixels": multiscale_pixels,
                        "multiscale_masks": multiscale_masks,
                        "multiscale_topk": self.multiscale_topk,
                        "multiscale_target_sizes": self.multiscale_target_sizes,
                    }

                stopping = self._uhrbat.KeywordsStoppingCriteria(stop_strings, self.tokenizer, input_ids)
                local_kwargs = dict(gen_kwargs)
                local_kwargs["stopping_criteria"] = [stopping]

                with torch.inference_mode():
                    output = self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **multimodal_kwargs,
                        **local_kwargs,
                    )

                scores = getattr(output, "scores", None)
                gen_steps = len(scores) if scores is not None else 0
                prompt_len = output.sequences.shape[1] - gen_steps if gen_steps else input_ids.shape[1]
                answer_ids = output.sequences[:, prompt_len:]
                text = _clean_generated_text(self._safe_decode(answer_ids))
                for stop in stop_strings:
                    if stop and stop in text:
                        text = text.split(stop)[0].strip()
                res.append(text)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)

            pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res: List[Tuple[float, bool]] = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            continuation = doc_to_target if isinstance(doc_to_target, str) else doc_to_target(doc)
            context = contexts[0] if isinstance(contexts, list) else contexts
            visuals = self._coerce_visuals(self._flatten([doc_to_visual(doc)]))

            context_prompt = self._build_prompt(context, len(visuals))
            full_prompt = self._build_prompt(context, len(visuals), continuation=str(continuation))
            context_ids = self._tokenize_prompt(context_prompt)
            input_ids = self._tokenize_prompt(full_prompt)
            labels = input_ids.clone()
            labels[:, : context_ids.shape[1]] = -100
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            multiscale_pixels, multiscale_masks, image_sizes = self._prepare_multiscale(visuals)
            multimodal_kwargs = {}
            if visuals:
                multimodal_kwargs = {
                    "image_sizes": image_sizes,
                    "modalities": ["image"],
                    "multiscale_pixels": multiscale_pixels,
                    "multiscale_masks": multiscale_masks,
                    "multiscale_topk": self.multiscale_topk,
                    "multiscale_target_sizes": self.multiscale_target_sizes,
                }

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=True,
                    **multimodal_kwargs,
                )

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, context_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, context_ids.shape[1] : input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("UHR-BAT does not implement multi-round generation in lmms-eval yet.")
