"""Score cached INSPIRE embeddings with the original cosine, Recall, and NDCG rules."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from iris.embed.experiments.inspire.data import (
    QUERY_COLUMNS,
    dataset_metadata,
    document_ids,
    load_inspire,
    read_run,
    write_run,
)

K_VALUES = (1, 5, 10, 20, 50, 100)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--k_values", nargs="+", type=int, default=list(K_VALUES))
    args = parser.parse_args()
    if any(k <= 0 for k in args.k_values):
        parser.error("--k_values must be positive")
    return args


def load_representations(directory, expected_ids):
    """Reject incomplete caches instead of silently scoring fewer candidates."""
    paths = list(directory.glob("*.npy"))
    found = {path.stem for path in paths}
    expected = set(expected_ids)
    if found != expected:
        raise ValueError(
            f"Incomplete or mismatched representations in {directory}: "
            f"{len(expected - found)} missing, {len(found - expected)} unexpected"
        )
    representations = {}
    shape = None
    # Preserve the original scorer's file enumeration and seeded tie-breaking.
    for path in tqdm(paths, desc="Loading representations"):
        representation = np.load(path, allow_pickle=False)
        if (
            representation.ndim != 1
            or not representation.size
            or not np.isfinite(representation).all()
        ):
            raise ValueError(f"Expected a finite, nonempty vector: {path}")
        if shape is not None and representation.shape != shape:
            raise ValueError(f"Inconsistent embedding dimensions: {path}")
        shape = representation.shape
        representations[path.stem] = representation
    return representations


def cosine_similarity(query_representation, document_representation):
    query_norm = np.linalg.norm(query_representation)
    document_norm = np.linalg.norm(document_representation)
    if query_norm == 0 or document_norm == 0:
        return 0.0
    return np.dot(query_representation, document_representation) / (
        query_norm * document_norm
    )


def compute_similarities(query_representation, document_representations, rng):
    similarities = [
        (
            document_id,
            cosine_similarity(query_representation, representation),
            rng.random(),
        )
        for document_id, representation in document_representations.items()
    ]
    similarities.sort(key=lambda item: (-item[1], item[2]))
    return [(document_id, similarity) for document_id, similarity, _ in similarities]


def calculate_recall_at_k(retrieved_ids, positive_ids, k):
    if not positive_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & positive_ids) / len(positive_ids)


def calculate_ndcg_at_k(retrieved_ids, positive_ids, k):
    if not positive_ids:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, document_id in enumerate(retrieved_ids[:k], start=1)
        if document_id in positive_ids
    )
    ideal_count = min(len(positive_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def score_query(query_id, query, query_representation, documents, k_values, rng):
    positive_ids = set(query["positive_documents"])
    excluded_ids = set(query["excluded_ids"])
    if not positive_ids <= documents.keys():
        raise ValueError(
            f"Query {query_id} has positives missing from the document set"
        )
    if positive_ids & excluded_ids:
        raise ValueError(f"Query {query_id} has excluded positive documents")
    candidates = {
        document_id: representation
        for document_id, representation in documents.items()
        if document_id not in excluded_ids
    }
    retrieved_ids = [
        document_id
        for document_id, _ in compute_similarities(
            query_representation, candidates, rng
        )
    ]
    return {
        "id": query_id,
        **{
            f"recall@{k}": float(calculate_recall_at_k(retrieved_ids, positive_ids, k))
            for k in k_values
        },
        **{
            f"ndcg@{k}": float(calculate_ndcg_at_k(retrieved_ids, positive_ids, k))
            for k in k_values
        },
        **{
            key: query[key]
            for key in QUERY_COLUMNS
            if key not in {"positive_documents", "excluded_ids"}
        },
        "top-100-retrieved-docs": retrieved_ids[:100],
    }


def score_split(run_dir, split, queries, documents, k_values, rng):
    representation_dir = run_dir / "representations"
    query_representations = load_representations(
        representation_dir / "queries" / split,
        [f"{index:05d}" for index in range(len(queries))],
    )
    document_representations = load_representations(
        representation_dir / "documents" / split, document_ids(documents)
    )
    if not query_representations or not document_representations:
        raise ValueError(f"Empty query or document set: {split}")
    if (
        next(iter(query_representations.values())).shape
        != next(iter(document_representations.values())).shape
    ):
        raise ValueError(f"Query and document embedding dimensions differ: {split}")
    path = run_dir / "scores" / split / "score.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for index, query in enumerate(tqdm(queries, desc=f"{split} scoring")):
            query_id = f"{index:05d}"
            entry = score_query(
                query_id,
                query,
                query_representations[query_id],
                document_representations,
                k_values,
                rng,
            )
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(
        f"[{split}] Scored {len(queries)} queries at {len(k_values)} k values: {path}"
    )


def main(args):
    metadata = read_run(args.run_dir)
    if not metadata["encoding_complete"]:
        raise ValueError("Encoding is incomplete; finish it before scoring")
    queries, documents = load_inspire(
        metadata["dataset_name"], metadata["dataset_revision"]
    )
    if dataset_metadata(queries, documents, metadata["splits"]) != metadata["dataset"]:
        raise ValueError("The dataset differs from the one used to encode this run")
    k_values = sorted(set(args.k_values))
    metadata["scoring"] = {"k_values": k_values, "complete": False}
    write_run(args.run_dir, metadata)
    rng = random.Random(metadata["seed"])
    for split in metadata["splits"]:
        score_split(
            args.run_dir,
            split,
            queries[split].select_columns(list(QUERY_COLUMNS)),
            documents[split].select_columns(["audio_id"]),
            k_values,
            rng,
        )
    metadata["scoring"]["complete"] = True
    write_run(args.run_dir, metadata)


if __name__ == "__main__":
    args = parse_args()
    main(args)
