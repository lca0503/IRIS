"""Load IRIS triplets with metadata for InfoNCE false-negative masking."""

import json
import os
from pathlib import Path

from datasets import Dataset
from swift.dataset import DatasetMeta, RowPreprocessor, SubsetDataset, register_dataset
from swift.dataset.loader import DatasetLoader

from iris.embed.modeling.embedding_prompt import embedding_text


class TextAudioRetrievePreprocessor(RowPreprocessor):
    """Keep query/positive/negative order and carry document metadata to the loss."""

    def preprocess(self, row):
        instruction = row["instruction"].strip()
        if not instruction:
            raise ValueError("instruction cannot be empty")
        criteria = {
            key: value
            for key, value in row["select_criteria"].items()
            if value is not None
        }
        if not criteria:
            raise ValueError("select_criteria must contain at least one non-None value")
        negatives = row["negative_documents"]
        negative_metadata = row["negative_documents_metadata"]
        if len(negatives) != len(negative_metadata):
            raise ValueError("Each negative document must have matching metadata")

        document_messages = [{"role": "user", "content": embedding_text()}]
        result = {
            "messages": [{"role": "user", "content": embedding_text(instruction)}],
            "audios": [row["spoken_query"]],
            "audio_weight_text": instruction,
            "positive_messages": [document_messages],
            "positive_audios": [[row["positive_document"]]],
            # JSON avoids Arrow null/string schema conflicts across source datasets.
            "select_criteria": json.dumps(criteria),
            "pos_doc_metadata": json.dumps(row["positive_document_metadata"]),
            "neg_doc_metadata": json.dumps(negative_metadata),
        }
        if negatives:
            result["negative_messages"] = [document_messages for _ in negatives]
            result["negative_audios"] = [[path] for path in negatives]
        return result


def resolve_audio_paths(row, audio_root):
    """Resolve local paths without changing the JSONL or its metadata."""

    def resolve(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else audio_root / path)

    return {
        **row,
        "spoken_query": resolve(row["spoken_query"]),
        "positive_document": resolve(row["positive_document"]),
        "negative_documents": [resolve(path) for path in row["negative_documents"]],
    }


class IRISEmbedDataLoader(DatasetLoader):
    """Read the complete JSONL before Arrow infers its nullable metadata schema."""

    def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf=None):
        subsets = getattr(dataset_syntax, "subsets", None) or ["train"]
        if len(subsets) != 1 or subsets[0] not in {"train", "val"}:
            raise ValueError("Use exactly one subset: iris:train or iris:val")
        variable = "IRIS_TRAIN_FILE" if subsets[0] == "train" else "IRIS_VAL_FILE"
        filename = os.environ.get(variable)
        if not filename:
            raise ValueError(f"Set {variable} to your JSONL path")
        metadata_file = Path(filename).expanduser().resolve(strict=True)
        audio_root = (
            Path(os.environ.get("IRIS_AUDIO_ROOT", metadata_file.parent))
            .expanduser()
            .resolve()
        )
        with metadata_file.open(encoding="utf-8") as file:
            rows = [
                resolve_audio_paths(json.loads(line), audio_root)
                for line in file
                if line.strip()
            ]
        if not rows:
            raise ValueError(f"Dataset is empty: {metadata_file}")
        dataset = Dataset.from_list(rows)
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset,
            num_proc=1,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
        )
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset


register_dataset(
    DatasetMeta(
        dataset_name="iris",
        loader=IRISEmbedDataLoader,
        preprocess_func=TextAudioRetrievePreprocessor(),
        subsets=[
            SubsetDataset(name="train", subset="default", split=["train"]),
            SubsetDataset(name="val", subset="default", split=["validation"]),
        ],
        tags=["embedding", "audio", "text"],
        huge_dataset=True,
    )
)
