"""Dataset and result metadata shared by the INSPIRE experiment scripts."""

import json
from pathlib import Path

# Keep the evaluation order used by the original seeded scoring loop.
EVALUATION_SPLITS = ("DailyTalk", "Expresso", "VCTK", "Synthetic")
REPORT_SPLITS = ("DailyTalk", "VCTK", "Expresso", "Synthetic")
QUERY_COLUMNS = (
    "positive_documents",
    "excluded_ids",
    "text",
    "title",
    "speaker",
    "speaking_style",
    "environmental_sound",
    "relevance",
    "instruction",
)


def load_inspire(dataset_name, revision=None):
    from datasets import load_dataset

    queries = load_dataset(dataset_name, "query", revision=revision)
    documents = load_dataset(dataset_name, "document", revision=revision)
    return queries, documents


def dataset_metadata(queries, documents, splits):
    """Record row counts and dataset fingerprints without decoding audio."""
    return {
        split: {
            "num_queries": len(queries[split]),
            "num_documents": len(documents[split]),
            "query_fingerprint": queries[split]._fingerprint,
            "document_fingerprint": documents[split]._fingerprint,
        }
        for split in splits
    }


def read_run(run_dir):
    return json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))


def write_run(run_dir, metadata):
    path = Path(run_dir) / "run.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def document_ids(documents):
    ids = list(documents["audio_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Document audio_id values must be unique within each split")
    for audio_id in ids:
        if (
            not isinstance(audio_id, str)
            or not audio_id
            or Path(audio_id).name != audio_id
        ):
            raise ValueError(f"Invalid document audio_id: {audio_id!r}")
        if audio_id in {".", ".."}:
            raise ValueError(f"Invalid document audio_id: {audio_id!r}")
    return ids
