"""Shared instruction tokenization for the two IRIS processors."""

from .embedding_prompt import resolve_audio_weight_texts


class _InstructionRouterProcessorMixin:
    """Tokenize query instructions, or the document tag when instruction is omitted."""

    def __call__(self, *args, audio_weight_text=None, **kwargs):
        inputs = super().__call__(*args, **kwargs)
        if "input_features" not in inputs:
            return inputs

        batch_size = inputs["input_ids"].shape[0]
        texts = resolve_audio_weight_texts(
            audio_weight_text,
            batch_size,
            self.audio_weight_document_text,
        )

        encoded = self.instruction_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs["audio_weight_input_ids"] = encoded["input_ids"]
        inputs["audio_weight_attention_mask"] = encoded["attention_mask"]
        return inputs
