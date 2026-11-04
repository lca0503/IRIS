"""MTEB audio adapter using the IRIS checkpoint's general query/document prompt."""

import numpy as np
import torch
from mteb.models.abs_encoder import AbsEncoder
from mteb.types import PromptType

from iris.embed.experiments.maeb.tasks import QUERY_DOCUMENT_TASKS, TASK_INSTRUCTIONS
from iris.embed.inference import last_token_pooling, load_audio
from iris.embed.modeling.audio_weight_inputs import add_audio_weight_inputs
from iris.embed.modeling.embedding_prompt import embedding_conversation

MAX_AUDIO_SECONDS = 120
EMBEDDING_SIDES = ("auto", "query", "document")


def identity(batch):
    return batch


def is_query_input(task_name, prompt_type, embedding_side):
    """Retrieval and reranking always honor MTEB's actual query/corpus roles."""
    if embedding_side not in EMBEDDING_SIDES:
        raise ValueError(f"Unsupported embedding side: {embedding_side!r}")
    if task_name not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unsupported MAEB task: {task_name}")
    if task_name in QUERY_DOCUMENT_TASKS:
        if prompt_type not in (PromptType.query, PromptType.document):
            raise ValueError(f"{task_name} requires an explicit query/document role")
        return prompt_type == PromptType.query
    if embedding_side == "auto":
        return prompt_type != PromptType.document
    return embedding_side == "query"


def prepare_audio(audio, sampling_rate):
    """Downmix channels, resample, then keep the first 120 seconds, as in MAEB."""
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array, source_rate = samples.data.numpy(), samples.sample_rate
    else:
        array, source_rate = np.asarray(audio["array"]), audio["sampling_rate"]
    if array.ndim > 1:
        array = array.mean(axis=0)
    waveform = load_audio({"array": array, "sampling_rate": source_rate}, sampling_rate)
    return waveform[: sampling_rate * MAX_AUDIO_SECONDS]


class MAEBEncoder(AbsEncoder):
    """Retain the benchmark's batched encoding and last-valid-token pooling."""

    mteb_model_meta = None

    def __init__(self, embedder, embedding_side="query"):
        if embedding_side not in EMBEDDING_SIDES:
            raise ValueError(f"Unsupported embedding side: {embedding_side!r}")
        self.embedder = embedder
        self.embedding_side = embedding_side

    @torch.no_grad()
    def encode(
        self, inputs, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs
    ):
        inputs.collate_fn = identity
        is_query = is_query_input(task_metadata.name, prompt_type, self.embedding_side)
        instruction = TASK_INSTRUCTIONS[task_metadata.name] if is_query else None
        embedder = self.embedder
        processor, model = embedder.processor, embedder.model
        representations = []
        for batch in inputs:
            audios = [
                prepare_audio(item["audio"], embedder.sampling_rate) for item in batch
            ]
            prompts = [
                processor.apply_chat_template(
                    embedding_conversation(audio, instruction),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for audio in audios
            ]
            processor_kwargs = {
                "text": prompts,
                "audio": audios,
                "sampling_rate": embedder.sampling_rate,
                "return_tensors": "pt",
                "padding": True,
            }
            if embedder.processor_routes_audio_weights:
                processor_kwargs["audio_weight_text"] = (
                    [instruction] * len(audios) if is_query else None
                )
            model_inputs = processor(**processor_kwargs)
            if not embedder.processor_routes_audio_weights:
                add_audio_weight_inputs(
                    model_inputs,
                    embedder.audio_weight_tokenizer,
                    model.config,
                    [instruction] * len(audios),
                    is_query,
                )
            model_inputs = model_inputs.to(model.device)
            output = model(**model_inputs, output_hidden_states=True, return_dict=True)
            representations.append(
                last_token_pooling(
                    output.hidden_states[-1], model_inputs["attention_mask"]
                )
                .cpu()
                .float()
                .numpy()
            )
        if not representations:
            raise ValueError("MAEB supplied no audio to encode")
        embeddings = np.concatenate(representations)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-8, None)
