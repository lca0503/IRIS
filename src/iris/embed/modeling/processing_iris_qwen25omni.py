"""Qwen2.5-Omni processor with an independent instruction tokenizer."""

from transformers import Qwen2_5OmniProcessor
from transformers.processing_utils import ProcessorMixin

from .audio_layer_aggregation import DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT

# Explicit re-export keeps the shared prompt dependency in HF's custom-code cache.
from .embedding_prompt import resolve_audio_weight_texts as resolve_audio_weight_texts
from .processing_audio_weights import _InstructionRouterProcessorMixin


class IrisQwen2_5OmniProcessor(_InstructionRouterProcessorMixin, Qwen2_5OmniProcessor):
    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        instruction_tokenizer=None,
        chat_template=None,
        audio_weight_document_text=DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT,
    ):
        ProcessorMixin.__init__(
            self,
            image_processor,
            video_processor,
            feature_extractor,
            tokenizer,
            instruction_tokenizer,
            chat_template=chat_template,
        )
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token
        self.audio_weight_document_text = audio_weight_document_text


IrisQwen2_5OmniProcessor.register_for_auto_class("AutoProcessor")
