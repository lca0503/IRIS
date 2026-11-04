# Copyright (c) IRIS Contributors. All rights reserved.
from transformers import AutoTokenizer

from .embedding_prompt import resolve_audio_weight_texts


def load_audio_weight_tokenizer(config, model_path=None, revision=None):
    """Load the bundled tokenizer, or the pinned MiniLM tokenizer for new training."""
    if (
        getattr(config, "audio_layer_aggregation", "final") != "learned"
        or getattr(config, "audio_weight_backend", None) != "minilm"
    ):
        return None
    if model_path is not None:
        return AutoTokenizer.from_pretrained(
            str(model_path), subfolder="instruction_tokenizer", revision=revision
        )
    return AutoTokenizer.from_pretrained(
        config.audio_weight_text_model,
        revision=getattr(config, "audio_weight_text_revision", None),
    )


def add_audio_weight_inputs(inputs, tokenizer, config, instructions, is_query):
    """Attach router tokens in the same form used by the training collator."""
    if tokenizer is None:
        return inputs
    batch_size = inputs["input_ids"].shape[0]
    texts = resolve_audio_weight_texts(
        instructions if is_query else None,
        batch_size,
        config.audio_weight_document_text,
    )
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs["audio_weight_input_ids"] = encoded["input_ids"]
    inputs["audio_weight_attention_mask"] = encoded["attention_mask"]
    return inputs
