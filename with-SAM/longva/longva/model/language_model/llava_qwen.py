#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import transformers
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from longva.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from longva.utils import rank0_print

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        Qwen2Model.__init__(self, config)
        LlavaMetaModel.__init__(self, config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig
    # 避免从 checkpoint 加载时针对 scale_pos_residuals 的“unused keys”告警
    _keys_to_ignore_on_load_unexpected = [r"model\.scale_pos_residuals\..*"]

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # K_FIX = 144
        # self.regproxy = RegProxyCompressor(
        #     in_dim=3584,
        #     h=4,
        #     w=4,
        #     k_fix=K_FIX,
        # )
        # Initialize weights and apply final processing
        self.post_init()
        # 共享多尺度 MLP 到外层，避免属性缺失
        if not hasattr(self, "multiscale_token_mlp") and hasattr(self.model, "multiscale_token_mlp"):
            self.multiscale_token_mlp = self.model.multiscale_token_mlp

    def load_state_dict(self, state_dict, strict: bool = False):
        """
        Be lenient when resuming from checkpoints that may be missing
        multiscale positional residuals introduced by newer code paths.
        """
        base = self.get_model()
        # Ensure container exists
        if not hasattr(base, "scale_pos_residuals"):
            base.scale_pos_residuals = nn.ParameterDict()

        # Ingest scale_pos_residuals from state_dict to avoid "unused keys" warnings
        consumed_keys = []
        for k, v in list(state_dict.items()):
            if not k.startswith("model.scale_pos_residuals."):
                continue
            parts = k.split(".")
            scale_key = parts[2] if len(parts) >= 3 else None
            if scale_key is None:
                continue
            if scale_key not in base.scale_pos_residuals:
                base.scale_pos_residuals[scale_key] = nn.Parameter(torch.zeros_like(v))
            with torch.no_grad():
                base.scale_pos_residuals[scale_key].data.copy_(v)
            consumed_keys.append(k)
        for k in consumed_keys:
            state_dict.pop(k, None)
        if consumed_keys:
            # 已从 checkpoint 恢复了残差，避免后续初始化覆盖
            setattr(base, "_scale_residual_initialized", True)

        # Always perform a lenient load so newly added modules fall back to fresh init.
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        error_msgs = list(getattr(incompatible, "error_msgs", []))

        def _is_scale_residual_key(k: str) -> bool:
            return k.startswith("model.scale_pos_residuals.") or k.startswith("scale_pos_residuals.")

        missing = [k for k in missing if not _is_scale_residual_key(k)]
        unexpected = [k for k in unexpected if not _is_scale_residual_key(k)]

        # Surface unrelated issues if strict was requested
        if strict and (missing or unexpected or error_msgs):
            messages = list(error_msgs)
            if missing:
                messages.append(f"Missing key(s) in state_dict: {', '.join(missing)}")
            if unexpected:
                messages.append(f"Unexpected key(s) in state_dict: {', '.join(unexpected)}")
            joined = "\n\t".join(messages)
            raise RuntimeError(f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{joined}")

        # Keep return type consistent (torch returns an IncompatibleKeys namedtuple)
        if hasattr(incompatible, "_replace"):
            try:
                return incompatible._replace(missing_keys=missing, unexpected_keys=unexpected)
            except Exception:
                pass
        return incompatible

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        vision_pruning: Optional[List] = None,
        multiscale_masks: Optional[Dict[int, torch.Tensor]] = None,
        multiscale_pixels: Optional[Dict[int, torch.Tensor]] = None,
        multiscale_topk: Optional[Sequence[int]] = None,
        multiscale_target_sizes: Optional[Sequence[int]] = None,
        use_scale_positional_residual: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 如果已经构造了 inputs_embeds，就不要再把 input_ids 传给基类，以免触发 "both input_ids and inputs_embeds" 的检查
        if inputs_embeds is not None:
            input_ids = None
        # 生成阶段（有 past_key_values）直接走缓存，不再重复多模态处理
        if past_key_values is not None and inputs_embeds is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if inputs_embeds is None:
            use_multiscale = multiscale_masks is not None and multiscale_topk is not None
            if use_multiscale:
                target_sizes = (
                    [int(s) for s in multiscale_target_sizes]
                    if multiscale_target_sizes is not None
                    else getattr(self.config, "multiscale_target_sizes", [672, 1344, 2688, 4032])
                )

                def _split_topk(topk_list, num_imgs: int):
                    per_image = []
                    for k in topk_list:
                        k = int(k)
                        base = k // num_imgs
                        extra = k % num_imgs
                        per_image.append([base + (1 if idx < extra else 0) for idx in range(num_imgs)])
                    # per scale -> per image; convert to per image -> per scale
                    return [[per_image[s][i] for s in range(len(topk_list))] for i in range(num_imgs)]

                masks_list = multiscale_masks if isinstance(multiscale_masks, list) else [multiscale_masks]
                pixels_list = multiscale_pixels if isinstance(multiscale_pixels, list) else [multiscale_pixels]
                num_imgs = max(len(masks_list), len(pixels_list))
                if num_imgs == 0:
                    num_imgs = 1
                topk_split = _split_topk(list(multiscale_topk), num_imgs)

                selected_features = []
                for img_idx in range(num_imgs):
                    mask_i = masks_list[img_idx] if img_idx < len(masks_list) else multiscale_masks
                    pixels_i = pixels_list[img_idx] if img_idx < len(pixels_list) else multiscale_pixels
                    # 是否启用尺度残差：默认关闭，如需开启可在调用时显式传 True 或在 config 中设置 enable_scale_pos_residual=True
                    add_scale_residual = use_scale_positional_residual or getattr(
                        self.config, "enable_scale_pos_residual", False
                    )
                    image_i = None
                    if pixels_i is None and images is not None:
                        if isinstance(images, list):
                            image_i = images[img_idx] if img_idx < len(images) else images[0]
                        else:
                            image_i = images

                    with torch.no_grad():
                        attn_info = self.compute_multiscale_image_attention(
                            image=pixels_i if pixels_i is not None else image_i,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            target_sizes=target_sizes,
                            position_ids=position_ids,
                            modalities=modalities,
                            num_images=1,
                            multiscale_masks=mask_i if isinstance(mask_i, dict) else None,
                        )
                    if getattr(self.config, "_log_vision_tokens", False):
                        try:
                            rank0_print(
                                f"[Debug] attn scales: {sorted(attn_info.get('attn_by_scale', {}).keys())}; "
                                f"mask scales: {sorted(int(k) for k in (mask_i or {}).keys()) if isinstance(mask_i, dict) else []}; "
                                f"topk split for image #{img_idx}: {topk_split[img_idx]}"
                            )
                        except Exception:
                            pass
                    embeds_by_scale, _, _ = self.encode_multiscale_tiles(
                        pixels_i if pixels_i is not None else image_i,
                        masks_by_scale=mask_i if isinstance(mask_i, dict) else None,
                        add_scale_residual=add_scale_residual,
                        add_tile_residual=False,
                    )
                    mask_clean = {}
                    if isinstance(mask_i, dict):
                        for k, v in mask_i.items():
                            if isinstance(v, dict) and "masks" in v:
                                mask_clean[int(k)] = v["masks"]
                    if not mask_clean:
                        mask_clean = mask_i if isinstance(mask_i, dict) else {}
                    selection = self.select_tokens_with_masks(
                        attn_info.get("attn_by_scale", {}),
                        embeds_by_scale,
                        mask_clean,
                        topk_split[img_idx],
                        target_sizes=target_sizes,
                        add_scale_residual=add_scale_residual,
                    )
                    # batch dim assumed 1
                    if not selection["selected_embeddings"] or selection["selected_embeddings"][0].numel() == 0:
                        rank0_print(
                            f"[Warn] No multiscale vision tokens selected for image #{img_idx}; using text-only for this sample."
                        )
                        selected_features.append(
                            torch.zeros(0, self.config.hidden_size, device=self.device, dtype=self.dtype)
                        )
                    else:
                        tokens_kept = selection["selected_embeddings"][0].shape[0]
                        rank0_print(
                            f"[Info] Selected {tokens_kept} multiscale vision tokens for image #{img_idx}."
                        )
                        if getattr(self.config, "_log_vision_tokens", False):
                            try:
                                scale_counts = []
                                for emb in selection.get("selected_embeddings", []):
                                    scale_counts.append(int(getattr(emb, "shape", [0])[0]) if emb is not None else 0)
                                rank0_print(f"[Debug] Image #{img_idx} per-scale selected counts: {scale_counts}")
                            except Exception:
                                pass
                        selected_features.append(selection["selected_embeddings"][0])

                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                ) = (
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images=None,
                        modalities=modalities,
                        image_sizes=image_sizes,
                        vision_pruning=vision_pruning,
                        image_features_override=selected_features,
                    )
                )
            else:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                    self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images,
                        modalities,
                        image_sizes,
                        vision_pruning=vision_pruning,
                    )
                )

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        multiscale_masks=None,
        multiscale_pixels=None,
        multiscale_topk=None,
        multiscale_target_sizes=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        orig_input_ids = inputs  # keep a copy so we can return full sequences
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        vision_pruning = kwargs.pop("vision_pruning", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        use_multiscale = multiscale_masks is not None and multiscale_topk is not None
        if use_multiscale:
            target_sizes = (
                [int(s) for s in multiscale_target_sizes]
                if multiscale_target_sizes is not None
                else getattr(self.config, "multiscale_target_sizes", [672, 1344, 2688, 4032])
            )
            masks_list = multiscale_masks if isinstance(multiscale_masks, list) else [multiscale_masks]
            pixels_list = multiscale_pixels if isinstance(multiscale_pixels, list) else [multiscale_pixels]
            num_imgs = max(len(masks_list), len(pixels_list))
            if num_imgs == 0:
                num_imgs = 1

            def _split_topk(topk_list, n_img: int):
                per_image = []
                for k in topk_list:
                    k = int(k)
                    base = k // n_img
                    extra = k % n_img
                    per_image.append([base + (1 if idx < extra else 0) for idx in range(n_img)])
                return [[per_image[s][i] for s in range(len(topk_list))] for i in range(n_img)]

            topk_split = _split_topk(list(multiscale_topk), num_imgs)
            selected_features = []
            add_scale_residual = getattr(self.config, "enable_scale_pos_residual", False)
            for img_idx in range(num_imgs):
                mask_i = masks_list[img_idx] if img_idx < len(masks_list) else multiscale_masks
                pixels_i = pixels_list[img_idx] if img_idx < len(pixels_list) else multiscale_pixels
                image_i = None
                if pixels_i is None and images is not None:
                    if isinstance(images, list):
                        image_i = images[img_idx] if img_idx < len(images) else images[0]
                    else:
                        image_i = images
                with torch.no_grad():
                    attn_info = self.compute_multiscale_image_attention(
                        image=pixels_i if pixels_i is not None else image_i,
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        target_sizes=target_sizes,
                        position_ids=position_ids,
                        modalities=modalities,
                        num_images=1,
                        multiscale_masks=mask_i if isinstance(mask_i, dict) else None,
                    )
                if getattr(self.config, "_log_vision_tokens", False):
                    try:
                        rank0_print(
                            f"[Debug] attn scales: {sorted(attn_info.get('attn_by_scale', {}).keys())}; "
                            f"mask scales: {sorted(int(k) for k in (mask_i or {}).keys()) if isinstance(mask_i, dict) else []}; "
                            f"topk split for image #{img_idx}: {topk_split[img_idx]}"
                        )
                    except Exception:
                        pass
                embeds_by_scale, _, _ = self.encode_multiscale_tiles(
                    pixels_i if pixels_i is not None else image_i,
                    masks_by_scale=mask_i if isinstance(mask_i, dict) else None,
                    add_scale_residual=add_scale_residual,
                    add_tile_residual=False,
                )
                mask_clean = {}
                if isinstance(mask_i, dict):
                    for k, v in mask_i.items():
                        if isinstance(v, dict) and "masks" in v:
                            mask_clean[int(k)] = v["masks"]
                if not mask_clean:
                    mask_clean = mask_i if isinstance(mask_i, dict) else {}
                selection = self.select_tokens_with_masks(
                    attn_info.get("attn_by_scale", {}),
                    embeds_by_scale,
                    mask_clean,
                    topk_split[img_idx],
                    target_sizes=target_sizes,
                    add_scale_residual=add_scale_residual,
                )
                if not selection["selected_embeddings"] or selection["selected_embeddings"][0].numel() == 0:
                    rank0_print(
                        f"[Warn] No multiscale vision tokens selected for image #{img_idx}; using text-only for this sample."
                    )
                    selected_features.append(
                        torch.zeros(0, self.config.hidden_size, device=self.device, dtype=self.dtype)
                    )
                else:
                    tokens_kept = selection["selected_embeddings"][0].shape[0]
                    rank0_print(f"[Info] Selected {tokens_kept} multiscale vision tokens for image #{img_idx}.")
                    if getattr(self.config, "_log_vision_tokens", False):
                        try:
                            scale_counts = []
                            for emb in selection.get("selected_embeddings", []):
                                scale_counts.append(int(getattr(emb, "shape", [0])[0]) if emb is not None else 0)
                            rank0_print(f"[Debug] Image #{img_idx} per-scale selected counts: {scale_counts}")
                        except Exception:
                            pass
                    selected_features.append(selection["selected_embeddings"][0])

            (
                _input_ids_placeholder,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                mm_input_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=None,
                modalities=modalities,
                image_sizes=image_sizes,
                vision_pruning=vision_pruning,
                image_features_override=selected_features,
                return_mm_input_ids=True,
            )
        elif images is not None:
            (
                _input_ids_placeholder,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                mm_input_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                vision_pruning=vision_pruning,
                return_mm_input_ids=True,
            )
        else:
            mm_input_ids = orig_input_ids
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 优先使用包含图像 token 的 mm_input_ids，以便生成时返回完整序列
        mm_input_ids = mm_input_ids if mm_input_ids is not None else orig_input_ids

        return super().generate(
            input_ids=mm_input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        vision_pruning = kwargs.pop("vision_pruning", None)
        multiscale_pixels = kwargs.pop("multiscale_pixels", None)
        multiscale_masks = kwargs.pop("multiscale_masks", None)
        multiscale_topk = kwargs.pop("multiscale_topk", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if vision_pruning is not None:
            inputs["vision_pruning"] = vision_pruning
        if multiscale_pixels is not None:
            inputs["multiscale_pixels"] = multiscale_pixels
        if multiscale_masks is not None:
            inputs["multiscale_masks"] = multiscale_masks
        if multiscale_topk is not None:
            inputs["multiscale_topk"] = multiscale_topk
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
