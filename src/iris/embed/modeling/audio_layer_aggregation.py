# Copyright (c) IRIS Contributors. All rights reserved.
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

AudioLayerAggregation = Literal["final", "fixed", "mean", "learned"]
AudioWeightBackend = Literal["minilm", "global"]

DEFAULT_AUDIO_WEIGHT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT = "Document:"


@dataclass(frozen=True)
class AudioLayerAggregationConfig:
    mode: AudioLayerAggregation = "final"
    layer_index: int | None = None
    train_projection: bool = False
    weight_backend: AudioWeightBackend = "minilm"
    init_final_prob: float | None = None
    text_model: str = DEFAULT_AUDIO_WEIGHT_TEXT_MODEL
    text_revision: str | None = None
    text_config: dict[str, Any] | None = None
    train_text_encoder: bool = False
    document_text: str = DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT

    def validate(self, num_audio_layers: int) -> None:
        if self.mode not in {
            "final",
            "fixed",
            "mean",
            "learned",
        }:
            raise ValueError(f"Unsupported audio layer aggregation: {self.mode!r}")
        if self.mode == "learned" and self.weight_backend not in {"minilm", "global"}:
            raise ValueError(
                f"Unsupported audio weight backend: {self.weight_backend!r}"
            )
        if num_audio_layers <= 0:
            raise ValueError("num_audio_layers must be positive")
        if self.init_final_prob is not None:
            if self.mode != "learned" or self.weight_backend != "minilm":
                raise ValueError(
                    "audio_weight_init_final_prob requires learned MiniLM routing"
                )
            if not 0 < self.init_final_prob < 1 or num_audio_layers < 2:
                raise ValueError(
                    "audio_weight_init_final_prob must be between 0 and 1 "
                    "with at least two audio states"
                )
        if self.mode == "fixed" and (
            self.layer_index is None or not 0 <= self.layer_index < num_audio_layers - 1
        ):
            raise ValueError(
                "audio_layer_index must identify an encoder layer in "
                f"[0, {num_audio_layers - 2}]"
            )
        if not self.document_text:
            raise ValueError("audio_weight_document_text cannot be empty")

    @property
    def selected_state_index(self) -> int | None:
        """Map an ``encoder_N`` analysis label to its hidden-state stack index."""
        if self.mode != "fixed" or self.layer_index is None:
            return None
        return self.layer_index + 1


class GlobalAudioLayerWeightModule(nn.Module):
    """Learn one audio-layer mixture shared by every input."""

    def __init__(self, num_audio_layers: int):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_audio_layers))

    def reset_parameters(self, initializer_range: float = 0.02) -> None:
        del initializer_range
        nn.init.zeros_(self.layer_logits)

    def forward(self, batch_size: int) -> torch.Tensor:
        weights = self.layer_logits.softmax(dim=-1)
        return weights.unsqueeze(0).expand(batch_size, -1)


class MiniLMAudioLayerWeightModule(nn.Module):
    """Route audio layers from a pretrained sentence encoder."""

    def __init__(
        self,
        text_config: dict[str, Any],
        num_audio_layers: int,
        train_text_encoder: bool = False,
        init_final_prob: float | None = None,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        model_type = text_config["model_type"]
        encoder_config = AutoConfig.for_model(
            model_type,
            **{key: value for key, value in text_config.items() if key != "model_type"},
        )
        self.text_encoder = AutoModel.from_config(encoder_config)
        self.train_text_encoder = train_text_encoder
        self.init_final_prob = init_final_prob
        self.norm = nn.LayerNorm(encoder_config.hidden_size)
        self.output_projection = nn.Linear(encoder_config.hidden_size, num_audio_layers)
        self.reset_routing_head()
        self.text_encoder.requires_grad_(train_text_encoder)

    def reset_routing_head(self) -> None:
        """Initialize near uniform or a final-layer prior, with nonzero head weights."""
        self.norm.reset_parameters()
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output_projection.bias)
        if self.init_final_prob is not None:
            probability = self.init_final_prob
            other_states = self.output_projection.out_features - 1
            if not 0 < probability < 1 or other_states < 1:
                raise ValueError(
                    "Final-layer initialization needs 0 < p < 1 and at least two states"
                )
            # Small random logits retain instruction-dependent gradients around this prior.
            bias = math.log(probability * other_states / (1 - probability))
            nn.init.constant_(self.output_projection.bias[-1:], bias)

    def load_pretrained_encoder(
        self, model_name: str, revision: str | None = None
    ) -> None:
        """Load the pinned sentence encoder when initializing a new IRIS model."""
        from transformers import AutoModel

        pretrained = AutoModel.from_pretrained(model_name, revision=revision)
        self.text_encoder.load_state_dict(pretrained.state_dict(), strict=True)
        self.text_encoder.requires_grad_(self.train_text_encoder)
        self.reset_routing_head()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_text_encoder:
            self.text_encoder.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        context = torch.enable_grad() if self.train_text_encoder else torch.no_grad()
        with context:
            hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        logits = self.output_projection(self.norm(pooled))
        return logits.softmax(dim=-1)


class OnlineAudioLayerAggregator:
    """Accumulate audio encoder states without retaining a layer stack."""

    def __init__(
        self,
        mode: AudioLayerAggregation,
        weights: torch.Tensor | None = None,
        selected_state_index: int | None = None,
    ):
        self.mode = mode
        self.weights = weights
        self.selected_state_index = selected_state_index
        self.value = None

    def add(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        owner_indices: torch.Tensor | None = None,
    ) -> None:
        if self.mode == "final":
            self.value = hidden_states
            return
        if self.mode == "fixed":
            if layer_index == self.selected_state_index:
                self.value = hidden_states
            return
        if self.mode == "mean":
            coefficient = 1.0
        else:
            if self.weights is None:
                raise ValueError("Weighted audio aggregation requires layer weights")
            coefficient = self.weights[:, layer_index]
            if owner_indices is not None:
                coefficient = coefficient[owner_indices]
            while coefficient.ndim < hidden_states.ndim:
                coefficient = coefficient.unsqueeze(-1)
        contribution = (
            hidden_states if self.mode == "mean" else hidden_states * coefficient
        )
        self.value = contribution if self.value is None else self.value + contribution

    def result(self, num_audio_layers: int) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("No audio layer states were accumulated")
        if self.mode == "mean":
            return self.value / num_audio_layers
        return self.value
