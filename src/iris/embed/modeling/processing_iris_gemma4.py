"""Gemma4 processor with an independent instruction tokenizer."""

from transformers import Gemma4Processor

from .audio_layer_aggregation import DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT

# Explicit re-export keeps the shared prompt dependency in HF's custom-code cache.
from .embedding_prompt import resolve_audio_weight_texts as resolve_audio_weight_texts
from .processing_audio_weights import _InstructionRouterProcessorMixin


class IrisGemma4Processor(_InstructionRouterProcessorMixin, Gemma4Processor):
    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        instruction_tokenizer,
        chat_template=None,
        image_seq_length=280,
        audio_seq_length=750,
        audio_ms_per_token=40,
        audio_weight_document_text=DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT,
        **kwargs,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            instruction_tokenizer=instruction_tokenizer,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
            audio_seq_length=audio_seq_length,
            audio_ms_per_token=audio_ms_per_token,
            **kwargs,
        )
        self.audio_weight_document_text = audio_weight_document_text


IrisGemma4Processor.register_for_auto_class("AutoProcessor")
