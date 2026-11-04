from types import MethodType

import torch
from transformers import Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ModelOutputWithPast,
    create_causal_mask_mapping,
    create_masks_for_generate,
    torch_compilable_check,
)

# Explicit re-export keeps this dependency in Hugging Face's custom-code cache.
from .audio_layer_aggregation import (
    AudioLayerAggregationConfig as AudioLayerAggregationConfig,
)
from .audio_layer_backends import configure_gemma_audio_aggregation


def _fix_dead_mask_rows(mask_mapping):
    """Prevent SDPA NaNs for padded queries with no visible keys."""
    if not isinstance(mask_mapping, dict):
        return mask_mapping
    for key, mask in mask_mapping.items():
        if not isinstance(mask, torch.Tensor):
            continue
        if mask.dtype == torch.bool:
            mask_mapping[key] = mask | ~mask.any(dim=-1, keepdim=True)
        elif mask.is_floating_point():
            dead = (mask == torch.finfo(mask.dtype).min).all(dim=-1, keepdim=True)
            mask_mapping[key] = mask.masked_fill(dead, 0)
    return mask_mapping


def _patch_audio_forward(model):
    """Install the audio-safe Gemma forward used by training and HF inference."""

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        position_ids=None,
        past_key_values=None,
        mm_token_type_ids=None,
        inputs_embeds=None,
        use_cache=None,
        image_position_ids=None,
        video_position_ids=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds
        )
        multimodal_mask = image_mask | video_mask | audio_mask
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if self.config.get_text_config().hidden_size_per_layer_input:
            pad_embedding = self.language_model.embed_tokens.weight[
                self.config.text_config.pad_token_id
            ]
            llm_inputs_embeds = torch.where(
                multimodal_mask[..., None],
                pad_embedding.to(multimodal_mask.device).view(1, 1, -1),
                inputs_embeds,
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(
                llm_input_ids, llm_inputs_embeds
            )
        else:
            per_layer_inputs = None

        image_features = None
        audio_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, image_position_ids, return_dict=True
            ).pooler_output
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            expanded_mask = (
                image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            torch_compilable_check(
                inputs_embeds[expanded_mask].numel() == image_features.numel(),
                "Image features and image tokens do not match",
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded_mask.to(inputs_embeds.device),
                image_features.to(inputs_embeds.device),
            )

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_position_ids, return_dict=True
            ).pooler_output
            video_features = video_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            expanded_mask = (
                video_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            torch_compilable_check(
                inputs_embeds[expanded_mask].numel() == video_features.numel(),
                "Video features and video tokens do not match",
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded_mask.to(inputs_embeds.device),
                video_features.to(inputs_embeds.device),
            )

        if input_features is not None:
            if input_features_mask is None:
                raise ValueError("input_features_mask is required with input_features")
            audio_output = self.get_audio_features(
                input_features, input_features_mask, return_dict=True
            )
            audio_features = audio_output.pooler_output[audio_output.attention_mask]
            expanded_mask = (
                audio_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            torch_compilable_check(
                inputs_embeds[expanded_mask].numel() == audio_features.numel(),
                "Audio features and audio tokens do not match",
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded_mask.to(inputs_embeds.device),
                audio_features.to(inputs_embeds.device),
            )

        if position_ids is None:
            past_length = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_length
            )
            position_ids = position_ids.unsqueeze(0)

        causal_masks = attention_mask
        if not isinstance(causal_masks, dict):
            if self.config.get_text_config().use_bidirectional_attention == "vision":
                causal_masks = create_causal_mask_mapping(
                    self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                )
            else:
                causal_masks = create_masks_for_generate(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                )
        causal_masks = _fix_dead_mask_rows(causal_masks)
        kwargs.pop("return_dict", None)
        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return Gemma4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            audio_hidden_states=audio_features,
        )

    model.origin_forward = model.forward
    model.forward = MethodType(forward, model)


class IrisGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma 4 with configurable audio-layer aggregation."""

    def __init__(self, config):
        super().__init__(config)
        self.model.audio_tower.set_attn_implementation("sdpa")
        _patch_audio_forward(self.model)
        configure_gemma_audio_aggregation(self)


IrisGemma4ForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
