"""Encode INSPIRE queries and documents with a Hugging Face IRIS checkpoint."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import set_seed

from iris.embed.experiments.inspire.data import (
    EVALUATION_SPLITS,
    dataset_metadata,
    document_ids,
    load_inspire,
    write_run,
)
from iris.embed.inference import IRISEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name", required=True, help="HF ID or local merged checkpoint"
    )
    parser.add_argument("--revision", help="Model revision")
    parser.add_argument(
        "--label", help="Model label used in reports; defaults to the checkpoint name"
    )
    parser.add_argument("--dataset_name", default="lca0503/INSPIRE")
    parser.add_argument("--dataset_revision")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=EVALUATION_SPLITS,
        default=list(EVALUATION_SPLITS),
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="New experiment run directory"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def encode_split(embedder, queries, documents, output_dir, split):
    """Keep one raw float32 vector per query row and per document audio_id."""
    query_dir = output_dir / "queries" / split
    document_dir = output_dir / "documents" / split
    query_dir.mkdir(parents=True)
    document_dir.mkdir(parents=True)
    for index, query in enumerate(tqdm(queries, desc="queries")):
        audio = query["audio"]
        waveform = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        representation = embedder.encode_queries(
            waveform, query["instruction"], normalize=False
        )[0]
        np.save(query_dir / f"{index:05d}.npy", representation)
    for document in tqdm(documents, desc="documents"):
        audio = document["audio"]
        waveform = {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
        representation = embedder.encode_documents(waveform, normalize=False)[0]
        np.save(document_dir / f"{document['audio_id']}.npy", representation)


def main(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {args.output_dir}")
    splits = [split for split in EVALUATION_SPLITS if split in args.splits]
    set_seed(args.seed)
    queries, documents = load_inspire(args.dataset_name, args.dataset_revision)
    for split in splits:
        document_ids(documents[split])
    embedder = IRISEmbedder.from_pretrained(
        args.model_name, revision=args.revision, device=args.device
    )
    metadata = {
        "model_name": args.model_name,
        "model_revision": embedder.revision,
        "label": args.label or Path(args.model_name).name,
        "prompt": "general",
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "splits": splits,
        "dataset": dataset_metadata(queries, documents, splits),
        "seed": args.seed,
        "encoding_complete": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run(args.output_dir, metadata)
    for split in splits:
        print(
            f"[{split}] Encoding {len(queries[split])} queries and {len(documents[split])} documents"
        )
        encode_split(
            embedder,
            queries[split],
            documents[split],
            args.output_dir / "representations",
            split,
        )
    metadata["encoding_complete"] = True
    write_run(args.output_dir, metadata)
    print(f"Representations saved to {args.output_dir / 'representations'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
