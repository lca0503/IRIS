"""Encode query and document audio using a merged IRIS Hugging Face checkpoint."""

from collections.abc import Mapping
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .modeling.audio_weight_inputs import (
    add_audio_weight_inputs,
    load_audio_weight_tokenizer,
)
from .modeling.embedding_prompt import embedding_conversation, query_router_text


def load_audio(audio, sampling_rate):
    """Read a file or a mono array/sampling_rate mapping, without duration truncation."""
    if isinstance(audio, (str, Path)):
        array, _ = librosa.load(str(audio), sr=sampling_rate, mono=True)
    elif isinstance(audio, Mapping):
        array = np.asarray(audio["array"], dtype=np.float32)
        source_rate = audio["sampling_rate"]
        if array.ndim != 1:
            raise ValueError("Audio arrays must be mono, with shape (num_samples,)")
        if source_rate <= 0:
            raise ValueError("Audio sampling_rate must be positive")
        if source_rate != sampling_rate:
            array = librosa.resample(
                array, orig_sr=source_rate, target_sr=sampling_rate
            )
    else:
        raise TypeError("Audio must be a file path or an array/sampling_rate mapping")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Audio must be a nonempty, finite mono waveform")
    return array


def last_token_pooling(hidden_states, attention_mask):
    """Select the last non-padding token, including the chat-template assistant prefix."""
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("Hidden states and attention mask have incompatible shapes")
    mask = attention_mask.to(device=hidden_states.device, dtype=torch.long)
    if not mask.bool().any(dim=1).all():
        raise ValueError("Every input must contain at least one non-padding token")
    indices = mask.shape[1] - 1 - mask.flip(1).argmax(dim=1)
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, indices]


def audio_list(audios):
    if isinstance(audios, (str, Path, Mapping)):
        audios = [audios]
    else:
        audios = list(audios)
    if not audios:
        raise ValueError("Provide at least one audio input")
    return audios


class IRISEmbedder:
    """A single-audio-at-a-time interface; no training or benchmark dependencies."""

    def __init__(self, model, processor, model_name=None, revision=None):
        config = model.config
        if not getattr(config, "iris_embedding_model", False):
            raise ValueError(
                "Use a merged IRIS checkpoint, not a base model or adapter"
            )
        if getattr(config, "iris_embedding_prompt_type", "general") != "general":
            raise ValueError("Only the general IRIS embedding prompt is supported")
        mode = getattr(config, "audio_layer_aggregation", "final")
        backend = getattr(config, "audio_weight_backend", None)
        if mode not in {"final", "fixed", "mean", "learned"}:
            raise ValueError(f"Unsupported audio aggregation: {mode!r}")
        if mode == "learned" and backend not in {"minilm", "global"}:
            raise ValueError(f"Unsupported audio weight backend: {backend!r}")
        self.model = model.eval()
        self.processor = processor
        self.model_name = model_name
        self.revision = revision
        self.sampling_rate = processor.feature_extractor.sampling_rate
        self.processor_routes_audio_weights = hasattr(
            processor, "instruction_tokenizer"
        )
        self.audio_weight_tokenizer = (
            None
            if self.processor_routes_audio_weights
            else load_audio_weight_tokenizer(config, model_name, revision=revision)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        *,
        revision=None,
        device="auto",
        dtype="auto",
        attn_implementation=None,
    ):
        """Load a trusted HF repository (or a local merged checkpoint for validation)."""
        model_kwargs = {
            "revision": revision,
            "trust_remote_code": True,
            "device_map": device,
            "dtype": dtype,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        resolved_revision = getattr(model.config, "_commit_hash", None) or revision
        processor = AutoProcessor.from_pretrained(
            model_name, revision=resolved_revision, trust_remote_code=True
        )
        return cls(model, processor, model_name, resolved_revision)

    def encode_queries(self, audios, instructions, *, normalize=True):
        """Encode queries with one shared instruction or one instruction per audio."""
        audios = audio_list(audios)
        instructions = (
            [instructions] * len(audios)
            if isinstance(instructions, str)
            else list(instructions)
        )
        if len(audios) != len(instructions):
            raise ValueError("Provide one instruction per audio, or a shared string")
        for instruction in instructions:
            query_router_text(instruction)
        return self._encode(audios, instructions, normalize)

    def encode_documents(self, audios, *, normalize=True):
        """Encode documents with Document:, without a query instruction."""
        audios = audio_list(audios)
        return self._encode(audios, [None] * len(audios), normalize)

    @torch.inference_mode()
    def _encode(self, audios, instructions, normalize):
        representations = []
        for item, instruction in zip(audios, instructions, strict=True):
            audio = load_audio(item, self.sampling_rate)
            processor_kwargs = (
                {"audio_weight_text": instruction}
                if self.processor_routes_audio_weights
                else None
            )
            inputs = self.processor.apply_chat_template(
                embedding_conversation(audio, instruction),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
            )
            if not self.processor_routes_audio_weights:
                add_audio_weight_inputs(
                    inputs,
                    self.audio_weight_tokenizer,
                    self.model.config,
                    instruction,
                    instruction is not None,
                )
            inputs = inputs.to(self.model.device)
            output = self.model(**inputs, output_hidden_states=True, return_dict=True)
            representation = last_token_pooling(
                output.hidden_states[-1], inputs["attention_mask"]
            )
            representations.append(representation.cpu().float().numpy())
        embeddings = np.concatenate(representations)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-8, None)
        return embeddings
