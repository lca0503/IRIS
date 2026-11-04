"""Check an exported IRIS checkpoint, optionally encoding a query and document."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

from iris.embed.inference import IRISEmbedder
from iris.embed.modeling.embedding_prompt import embedding_conversation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--audio_file", type=Path)
    return parser.parse_args()


def checkpoint_keys(directory):
    index_file = directory / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        for filename in set(weight_map.values()):
            if not (directory / filename).is_file():
                raise ValueError(f"Missing weight shard: {filename}")
        return set(weight_map)
    keys = set()
    for filename in directory.glob("*.safetensors"):
        with safe_open(filename, framework="pt") as file:
            keys.update(file.keys())
    return keys


def validate_checkpoint(directory):
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    if not getattr(config, "iris_embedding_model", False):
        raise ValueError("The config does not identify an IRIS embedding model")
    mode = getattr(config, "audio_layer_aggregation", None)
    if mode not in {"final", "fixed", "mean", "learned"}:
        raise ValueError(f"Unsupported audio aggregation: {mode}")
    if not getattr(config, "audio_weight_includes_pre_encoder", False):
        raise ValueError("The audio-state configuration omits the pre-encoder state")
    if mode == "fixed":
        index = getattr(config, "audio_layer_index", None)
        states = getattr(config, "audio_weight_num_output_layers", 0)
        if type(index) is not int or not 0 <= index < states - 1:
            raise ValueError("Invalid zero-based audio encoder layer index")

    auto_map = getattr(config, "auto_map", {})
    if "AutoModelForCausalLM" not in auto_map:
        raise ValueError("The config does not register AutoModelForCausalLM")
    modeling_file = auto_map["AutoModelForCausalLM"].split(".", 1)[0] + ".py"
    for filename in (
        modeling_file,
        "audio_layer_backends.py",
        "audio_layer_aggregation.py",
    ):
        if not (directory / filename).is_file():
            raise ValueError(f"Missing custom model file: {filename}")

    keys = checkpoint_keys(directory)
    if not keys:
        raise ValueError("No safetensors model weights found")
    if any("lora_A" in key or "lora_B" in key for key in keys):
        raise ValueError("Unmerged LoRA weights remain in the checkpoint")
    minilm = mode == "learned" and config.audio_weight_backend == "minilm"
    if mode == "learned":
        modules = (
            ["text_encoder", "norm", "output_projection"]
            if minilm
            else ["layer_logits"]
        )
        for module in modules:
            prefix = f"audio_layer_weight_module.{module}"
            if not any(key.startswith(prefix) for key in keys):
                raise ValueError(f"Missing trained module: {prefix}")

    processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
    if minilm and not hasattr(processor, "instruction_tokenizer"):
        raise ValueError("AutoProcessor did not load the bundled MiniLM tokenizer")
    family = "qwen" if "qwen" in config.architectures[0].lower() else "gemma"
    suffix = "<|im_start|>assistant\n" if family == "qwen" else "<|turn>model\n"
    for instruction in (None, "Same speaker."):
        rendered = processor.apply_chat_template(
            embedding_conversation("audio.wav", instruction),
            add_generation_prompt=True,
            tokenize=False,
        )
        if "system" in rendered.lower() or not rendered.endswith(suffix):
            raise ValueError("The chat template does not match embedding training")
        expected = (
            "Document:" if instruction is None else "Instruction: Same speaker.\nQuery:"
        )
        if expected not in rendered:
            raise ValueError("The chat template lost the query/document prefix")
    return config


def main(args):
    config = validate_checkpoint(args.checkpoint_dir)
    print(f"Checkpoint and processor are valid: {config.architectures[0]}")
    if args.audio_file is None:
        return
    model = IRISEmbedder.from_pretrained(str(args.checkpoint_dir))
    queries = model.encode_queries([args.audio_file], instructions="Same speaker.")
    documents = model.encode_documents([args.audio_file])
    if queries.shape != documents.shape or not np.isfinite([queries, documents]).all():
        raise ValueError("Query/document inference returned invalid embeddings")
    print(f"Query/document inference passed: embedding shape {queries.shape}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
