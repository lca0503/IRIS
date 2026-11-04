from transformers import Qwen2_5OmniThinkerForConditionalGeneration

# Explicit re-export keeps this dependency in Hugging Face's custom-code cache.
from .audio_layer_aggregation import (
    AudioLayerAggregationConfig as AudioLayerAggregationConfig,
)
from .audio_layer_backends import configure_qwen_audio_aggregation


class IrisQwen2_5OmniThinkerForConditionalGeneration(
    Qwen2_5OmniThinkerForConditionalGeneration
):
    """Qwen2.5-Omni Thinker with configurable audio-layer aggregation."""

    def __init__(self, config):
        super().__init__(config)
        configure_qwen_audio_aggregation(self)


IrisQwen2_5OmniThinkerForConditionalGeneration.register_for_auto_class(
    "AutoModelForCausalLM"
)
