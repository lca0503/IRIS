"""Finalize an IRIS checkpoint for standalone Hugging Face inference."""

import argparse
from pathlib import Path

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from iris.embed.modeling.processing_iris_gemma4 import IrisGemma4Processor
from iris.embed.modeling.processing_iris_qwen25omni import IrisQwen2_5OmniProcessor
from iris.embed.modeling.audio_layer_aggregation import DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT

QWEN_CHAT_TEMPLATE = """{% set audio_count = namespace(value=0) %}{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_bos|><|IMAGE|><|vision_eos|>{% elif content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if add_audio_id %}Audio {{ audio_count.value }}: {% endif %}<|audio_bos|><|AUDIO|><|audio_eos|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_bos|><|VIDEO|><|vision_eos|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

GEMMA_CHAT_TEMPLATE = """<bos>{% for message in messages %}<|turn>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<turn|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'audio' or 'audio' in content or 'audio_url' in content %}<|audio|>{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}<|image|>{% elif content['type'] == 'video' or 'video' in content %}<|video|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<turn|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|turn>model
{% endif %}"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    return parser.parse_args()


def model_family(config):
    architecture = " ".join(getattr(config, "architectures", [])).lower()
    if "qwen" in architecture:
        return "qwen"
    if "gemma" in architecture:
        return "gemma"
    raise ValueError(f"Unsupported IRIS architecture: {config.architectures}")


def instruction_aware_processor(processor, config, family):
    instruction_tokenizer = AutoTokenizer.from_pretrained(
        config.audio_weight_text_model,
        revision=getattr(config, "audio_weight_text_revision", None),
    )
    common = {
        "feature_extractor": processor.feature_extractor,
        "image_processor": processor.image_processor,
        "video_processor": processor.video_processor,
        "tokenizer": processor.tokenizer,
        "instruction_tokenizer": instruction_tokenizer,
        "chat_template": processor.chat_template,
        "audio_weight_document_text": getattr(
            config,
            "audio_weight_document_text",
            DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT,
        ),
    }
    if family == "qwen":
        return IrisQwen2_5OmniProcessor(**common)
    return IrisGemma4Processor(
        **common,
        image_seq_length=processor.image_seq_length,
        audio_seq_length=processor.audio_seq_length,
        audio_ms_per_token=processor.audio_ms_per_token,
    )


def main(args):
    config = AutoConfig.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    family = model_family(config)
    processor = AutoProcessor.from_pretrained(
        args.checkpoint_dir, trust_remote_code=True
    )
    template = QWEN_CHAT_TEMPLATE if family == "qwen" else GEMMA_CHAT_TEMPLATE
    processor.chat_template = template
    processor.tokenizer.chat_template = template
    if (
        config.audio_layer_aggregation == "learned"
        and config.audio_weight_backend == "minilm"
    ):
        processor = instruction_aware_processor(processor, config, family)
    processor.save_pretrained(args.checkpoint_dir)
    print(f"Prepared standalone Hugging Face processor in {args.checkpoint_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
