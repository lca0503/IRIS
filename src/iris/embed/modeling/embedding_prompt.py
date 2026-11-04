# Copyright (c) IRIS Contributors. All rights reserved.
from .audio_layer_aggregation import DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT


def query_router_text(instruction):
    """Return the query text seen by an independent audio-layer router."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    return f"Instruction: {instruction}\nQuery:"


def resolve_audio_weight_texts(instructions, batch_size, document_text):
    """Resolve one canonical MiniLM routing string per audio input."""
    if instructions is None:
        texts = [document_text] * batch_size
    else:
        instructions = (
            [instructions] * batch_size
            if isinstance(instructions, str)
            else list(instructions)
        )
        if len(instructions) != batch_size:
            raise ValueError("audio_weight_text must match the input batch size")
        texts = [query_router_text(instruction) for instruction in instructions]
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("audio_weight_text entries must be non-empty strings")
    return texts


def embedding_prefix(instruction=None):
    """Return the canonical text placed immediately before query/document audio."""
    if instruction is None:
        return f"{DEFAULT_AUDIO_WEIGHT_DOCUMENT_TEXT} "
    return f"{query_router_text(instruction)} "


def embedding_text(instruction=None):
    """Return the canonical training text containing the audio placeholder."""
    return f"{embedding_prefix(instruction)}<audio>"


def embedding_conversation(audio, instruction=None):
    """Return the canonical no-system inference conversation."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": embedding_prefix(instruction)},
                {"type": "audio", "audio": audio},
            ],
        }
    ]
