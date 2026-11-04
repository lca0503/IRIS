"""Replace query instructions while reusing one INSPIRE run's document embeddings."""

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from iris.embed.experiments.inspire.data import (
    EVALUATION_SPLITS,
    dataset_metadata,
    document_ids,
    load_inspire,
    read_run,
    write_run,
)
from iris.embed.experiments.inspire.score import K_VALUES, load_representations, score_query
from iris.embed.inference import IRISEmbedder, last_token_pooling, load_audio

EMPTY_QUERY_TEXT = "Instruction: \nQuery:"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_run_dir",
        type=Path,
        required=True,
        help="Completed INSPIRE encoding run for this checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="New instruction-ablation directory",
    )
    parser.add_argument(
        "--instructions",
        nargs="+",
        required=True,
        help='Exact replacement instructions; use "" for no instruction',
    )
    parser.add_argument(
        "--relevances",
        nargs="+",
        help="Exact dataset relevance labels; omit to include all queries",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=EVALUATION_SPLITS,
        help="Defaults to the base run's splits",
    )
    parser.add_argument("--k_values", nargs="+", type=int, default=list(K_VALUES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(k <= 0 for k in args.k_values):
        parser.error("--k_values must be positive")
    if args.relevances is not None and any(
        not value.strip() for value in args.relevances
    ):
        parser.error("Omit --relevances to include all queries")
    return args


def named_conditions(values, empty_name):
    """Keep exact text and reject ambiguous directory names."""
    conditions = []
    for value in values:
        if value and not value.strip():
            raise ValueError('Use exactly "" rather than whitespace for no instruction')
        name = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
        conditions.append((name or empty_name, value))
    names = [name for name, _ in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Conditions produce duplicate result directory names")
    return conditions


def select_queries(queries, splits, relevance):
    selected, indices = {}, {}
    for split in splits:
        labels = queries[split]["relevance"]
        matches = [
            index
            for index, value in enumerate(labels)
            if relevance is None or value == relevance
        ]
        if matches:
            selected[split] = queries[split].select(matches)
            indices[split] = matches
    if not selected:
        raise ValueError(
            f"No queries matched relevance {relevance!r} in the requested splits"
        )
    return selected, indices


@torch.inference_mode()
def encode_query(embedder, query, instruction):
    """Keep the empty-instruction control local to the experiment."""
    audio = query["audio"]
    waveform = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
    if instruction:
        return embedder.encode_queries(waveform, instruction, normalize=False)[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{EMPTY_QUERY_TEXT} "},
                {
                    "type": "audio",
                    "audio": load_audio(waveform, embedder.sampling_rate),
                },
            ],
        }
    ]
    inputs = embedder.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"audio_weight_text": None}
        if embedder.processor_routes_audio_weights
        else None,
    )
    tokenizer = (
        embedder.processor.instruction_tokenizer
        if embedder.processor_routes_audio_weights
        else embedder.audio_weight_tokenizer
    )
    if tokenizer is not None:
        routed = tokenizer(
            [EMPTY_QUERY_TEXT], padding=True, truncation=True, return_tensors="pt"
        )
        inputs["audio_weight_input_ids"] = routed["input_ids"]
        inputs["audio_weight_attention_mask"] = routed["attention_mask"]
    inputs = inputs.to(embedder.model.device)
    output = embedder.model(**inputs, output_hidden_states=True, return_dict=True)
    return (
        last_token_pooling(output.hidden_states[-1], inputs["attention_mask"])[0]
        .cpu()
        .float()
        .numpy()
    )


def evaluate_condition(
    embedder, queries, documents, instruction, run_dir, k_values, seed
):
    """Use dataset document order and reset tie-breaking per split and condition."""
    for split, split_queries in queries.items():
        query_dir = run_dir / "representations" / "queries" / split
        query_dir.mkdir(parents=True)
        score_file = run_dir / "scores" / split / "score.jsonl"
        score_file.parent.mkdir(parents=True)
        temporary = score_file.with_suffix(".jsonl.tmp")
        rng = random.Random(f"{seed}:{split}")
        dimension = next(iter(documents[split].values())).shape
        with temporary.open("w", encoding="utf-8") as file:
            for index, query in enumerate(
                tqdm(split_queries, desc=f"{split}: {instruction or 'no instruction'}")
            ):
                representation = encode_query(embedder, query, instruction)
                if (
                    representation.shape != dimension
                    or not np.isfinite(representation).all()
                ):
                    raise ValueError(f"Invalid query embedding in {split}, row {index}")
                query_id = f"{index:05d}"
                np.save(query_dir / f"{query_id}.npy", representation)
                entry = score_query(
                    query_id, query, representation, documents[split], k_values, rng
                )
                entry["original_instruction"] = entry["instruction"]
                entry["instruction"] = instruction
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        temporary.replace(score_file)
        print(
            f"[{split}/{run_dir.name}] Scored {len(split_queries)} queries at {len(k_values)} k values"
        )
    metadata = read_run(run_dir)
    metadata["encoding_complete"] = True
    metadata["scoring"]["complete"] = True
    write_run(run_dir, metadata)


def main(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {args.output_dir}")
    base = read_run(args.base_run_dir)
    if not base["encoding_complete"] or base.get("analysis_type"):
        raise ValueError("--base_run_dir must be a completed main INSPIRE encoding run")
    conditions = named_conditions(args.instructions, "no_instruction")
    relevances = (
        named_conditions(args.relevances, "all") if args.relevances else [("all", None)]
    )
    splits = args.splits or base["splits"]
    if not set(splits) <= set(base["splits"]):
        raise ValueError(
            "Requested splits must have document embeddings in the base run"
        )
    splits = [split for split in EVALUATION_SPLITS if split in splits]
    k_values = sorted(set(args.k_values))
    set_seed(args.seed)
    queries, documents = load_inspire(base["dataset_name"], base["dataset_revision"])
    if dataset_metadata(queries, documents, splits) != {
        split: base["dataset"][split] for split in splits
    }:
        raise ValueError("The dataset differs from the base INSPIRE run")
    selections = {
        name: (relevance, *select_queries(queries, splits, relevance))
        for name, relevance in relevances
    }
    used_splits = [
        split
        for split in splits
        if any(split in selected for _, selected, _ in selections.values())
    ]
    cached_documents = {}
    for split in used_splits:
        ids = document_ids(documents[split])
        if not ids:
            raise ValueError(f"Empty document set: {split}")
        cached = load_representations(
            args.base_run_dir / "representations" / "documents" / split, ids
        )
        cached_documents[split] = {audio_id: cached[audio_id] for audio_id in ids}
    embedder = IRISEmbedder.from_pretrained(
        base["model_name"], revision=base["model_revision"], device=args.device
    )
    if embedder.revision != base["model_revision"]:
        raise ValueError(
            "The loaded checkpoint revision differs from the document encoding run"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        f"{relevance_name}/{name}"
        for relevance_name in selections
        for name, _ in conditions
    ]
    index = {
        "analysis_type": "instruction_ablation",
        "source_run": os.path.relpath(
            args.base_run_dir.resolve(), args.output_dir.resolve()
        ),
        "model_name": base["model_name"],
        "model_revision": base["model_revision"],
        "runs": runs,
        "complete": False,
    }
    write_run(args.output_dir, index)
    for relevance_name, (relevance, selected, indices) in selections.items():
        for name, instruction in conditions:
            run_dir = args.output_dir / relevance_name / name
            run_dir.mkdir(parents=True)
            metadata = {
                **base,
                "analysis_type": "instruction_ablation",
                "source_run": os.path.relpath(
                    args.base_run_dir.resolve(), run_dir.resolve()
                ),
                "instruction": instruction,
                "relevance_filter": relevance,
                "query_indices": indices,
                "splits": list(selected),
                "dataset": dataset_metadata(selected, documents, selected),
                "seed": args.seed,
                "encoding_complete": False,
                "scoring": {"k_values": k_values, "complete": False},
            }
            write_run(run_dir, metadata)
            evaluate_condition(
                embedder,
                selected,
                cached_documents,
                instruction,
                run_dir,
                k_values,
                args.seed,
            )
    index["complete"] = True
    write_run(args.output_dir, index)
    print(f"Instruction-ablation results saved to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
