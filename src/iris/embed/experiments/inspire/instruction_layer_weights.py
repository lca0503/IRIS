"""Export raw router weights and RMS-adjusted weights on shared INSPIRE audio."""

import argparse
import copy
import csv
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from iris.embed.experiments.inspire.data import write_run
from iris.embed.inference import IRISEmbedder, load_audio
from iris.embed.modeling.audio_weight_inputs import add_audio_weight_inputs
from iris.embed.modeling.embedding_prompt import embedding_conversation, query_router_text

SPLIT_COUNTS = {"DailyTalk": 200, "VCTK": 80, "Expresso": 200, "Synthetic": 200}
DATASET_REVISION = "4202b965891ca79897ce8cb639e08c591e993c06"
INSTRUCTIONS = (
    (
        "contain answer",
        "Find the answer.",
        "Find speech that contains the answer to the question in the query.",
    ),
    (
        "continuation",
        "Continuation.",
        "Find speech that continues the conversation in the query.",
    ),
    (
        "same speaker",
        "Same speaker.",
        "Find speech from the same speaker as the query.",
    ),
    (
        "same speaking style",
        "Same speaking style.",
        "Find speech with the same speaking style as the query.",
    ),
    (
        "same environmental sound",
        "Same environmental sound.",
        "Find speech with the same environmental sound as the query.",
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name", required=True, help="HF ID or local IRIS checkpoint"
    )
    parser.add_argument("--revision", help="Model revision")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--title", help="Backbone name displayed above the heatmap")
    parser.add_argument("--dataset_revision", default=DATASET_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights_only", action="store_true", help="Skip audio and RMS"
    )
    return parser.parse_args()


def conditions():
    rows = []
    for criterion, short, normal in INSTRUCTIONS:
        for form, instruction in (("short", short), ("normal", normal)):
            rows.append(
                {
                    "condition": f"{criterion.replace(' ', '_')}_{form}",
                    "role": "query",
                    "criterion": criterion,
                    "form": form,
                    "instruction": instruction,
                    "router_text": query_router_text(instruction),
                }
            )
    rows.append(
        {
            "condition": "document",
            "role": "document",
            "criterion": "",
            "form": "document",
            "instruction": "",
            "router_text": "Document:",
        }
    )
    return rows


def layer_names(num_states):
    return ["pre_encoder"] + [f"encoder_{index}" for index in range(num_states - 1)]


def audio_tower(model):
    if model.config.model_type == "gemma4":
        return model.model.audio_tower
    if model.config.model_type == "qwen2_5_omni_thinker":
        return model.audio_tower
    raise ValueError(f"Unsupported backbone: {model.config.model_type}")


def validate_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.ndim != 2
        or min(weights.shape) == 0
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.allclose(weights.sum(-1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Weights must be finite, nonnegative rows summing to one")
    return weights


@torch.inference_mode()
def extract_weights(embedder, rows):
    """Use an independent float32 CPU copy; leave inference parameters unchanged."""
    model, processor = embedder.model, embedder.processor
    config = model.config
    if (
        config.audio_layer_aggregation != "learned"
        or config.audio_weight_backend not in {"minilm", "global"}
        or not config.audio_weight_includes_pre_encoder
    ):
        raise ValueError(
            "This analysis requires MiniLM or global weights including pre_encoder"
        )
    if config.audio_weight_document_text != "Document:":
        raise ValueError("Expected the canonical Document: routing text")
    router = copy.deepcopy(model.audio_layer_weight_module).cpu().float().eval()
    if config.audio_weight_backend == "global":
        weights = router(len(rows)).numpy()
    else:
        tokenizer = getattr(processor, "instruction_tokenizer", None)
        if tokenizer is None or processor.audio_weight_document_text != "Document:":
            raise ValueError(
                "Checkpoint processor must supply the matching MiniLM tokenizer"
            )
        weights = []
        for row in rows:
            inputs = tokenizer(row["router_text"], truncation=True, return_tensors="pt")
            weights.append(
                router(inputs["input_ids"], inputs["attention_mask"])[0].numpy()
            )
        weights = np.stack(weights)
    if weights.shape != (len(rows), len(audio_tower(model).layers) + 1):
        raise ValueError(
            "Router outputs do not match pre_encoder plus the encoder layers"
        )
    return validate_weights(weights)


def query_pool(queries):
    """Keep the first row for each (split, audio_id); no random sampling."""
    samples = []
    for split, expected in SPLIT_COUNTS.items():
        seen = set()
        ids = queries[split].select_columns(["audio_id"])["audio_id"]
        for index, audio_id in enumerate(ids):
            if not isinstance(audio_id, str) or not audio_id:
                raise ValueError(f"{split}: query audio_id must be a nonempty string")
            if audio_id not in seen:
                samples.append(
                    {"split": split, "dataset_index": index, "audio_id": audio_id}
                )
                seen.add(audio_id)
        if len(seen) != expected:
            raise ValueError(
                f"{split}: expected {expected} unique query audios, found {len(seen)}"
            )
    return samples


class FrameRMS:
    """Observe raw encoder states without changing their outputs or saving frames."""

    def __init__(self, tower):
        self.num_states = len(tower.layers) + 1
        self.values = []
        self.handles = [
            tower.layers[0].register_forward_pre_hook(
                self.pre_encoder, with_kwargs=True
            )
        ]
        self.handles.extend(
            layer.register_forward_hook(self.encoder) for layer in tower.layers
        )

    def add(self, hidden):
        if hidden.ndim == 3 and hidden.shape[0] == 1:
            hidden = hidden[0]
        if hidden.ndim != 2 or min(hidden.shape) == 0:
            raise ValueError("Expected one audio's [frames, channels] states")
        variance, mean = torch.var_mean(hidden.detach().float(), dim=-1, correction=0)
        self.values.append((variance + mean.square()).sqrt().cpu().numpy())

    def pre_encoder(self, module, inputs, kwargs):
        hidden = kwargs.get("hidden_states")
        self.add(inputs[0] if hidden is None else hidden)

    def encoder(self, module, inputs, output):
        self.add(output[0] if isinstance(output, tuple) else output)

    def result(self, valid_frames):
        valid_frames = np.asarray(valid_frames, dtype=bool)
        if valid_frames.ndim != 1 or not valid_frames.any():
            raise ValueError("Expected a nonempty valid-frame mask")
        if len(self.values) != self.num_states or any(
            value.shape != valid_frames.shape for value in self.values
        ):
            raise ValueError(
                "Captured encoder states do not match the audio frame mask"
            )
        values = np.stack(self.values)[:, valid_frames]
        if not np.isfinite(values).all():
            raise ValueError("Audio RMS contains non-finite values")
        return values.mean(axis=1, dtype=np.float64), int(valid_frames.sum())

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def encode_rms(embedder, tap, audio):
    """Run the checkpoint's processor and audio encoder, without an LLM forward."""
    waveform = load_audio(audio, embedder.sampling_rate)
    inputs = embedder.processor.apply_chat_template(
        embedding_conversation(waveform),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if not embedder.processor_routes_audio_weights:
        add_audio_weight_inputs(
            inputs, embedder.audio_weight_tokenizer, embedder.model.config, None, False
        )
    model = embedder.model
    tower = audio_tower(model)
    parameter = next(tower.parameters())
    inputs = inputs.to(parameter.device)
    model.prepare_audio_layer_weights(
        inputs["input_ids"],
        inputs["attention_mask"],
        audio_weight_input_ids=inputs.get("audio_weight_input_ids"),
        audio_weight_attention_mask=inputs.get("audio_weight_attention_mask"),
    )
    tap.values.clear()
    features = inputs["input_features"].to(dtype=parameter.dtype)
    if model.config.model_type == "gemma4":
        output = tower(
            input_features=features,
            attention_mask=inputs["input_features_mask"],
            return_dict=True,
        )
        valid_frames = output.attention_mask[0].detach().bool().cpu().numpy()
    else:
        model.get_audio_features(features, inputs["feature_attention_mask"])
        lengths, _ = tower._get_feat_extract_output_lengths(
            inputs["feature_attention_mask"].sum(1)
        )
        valid_frames = np.ones(int(lengths.item()), dtype=bool)
    return tap.result(valid_frames)


def measure_rms(embedder, queries, samples, names):
    from datasets import Audio

    records = []
    with closing(FrameRMS(audio_tower(embedder.model))) as tap:
        for split in SPLIT_COUNTS:
            dataset = queries[split].cast_column(
                "audio", Audio(sampling_rate=embedder.sampling_rate)
            )
            selected = [sample for sample in samples if sample["split"] == split]
            for sample in tqdm(selected, desc=f"RMS/{split}"):
                item = dataset[sample["dataset_index"]]
                if item["audio_id"] != sample["audio_id"]:
                    raise ValueError("Audio metadata no longer matches the saved pool")
                audio = item["audio"]
                rms, num_frames = encode_rms(
                    embedder,
                    tap,
                    {
                        "array": audio["array"],
                        "sampling_rate": audio["sampling_rate"],
                    },
                )
                records.append(
                    {
                        **sample,
                        "num_frames": num_frames,
                        **dict(zip(names, rms, strict=True)),
                    }
                )
    return records


def adjust_weights(weights, rms_by_split):
    """Normalize weight × RMS per audio, then average within and across splits."""
    weights = validate_weights(weights)
    if not rms_by_split:
        raise ValueError("No audio RMS supplied")
    adjusted = {}
    for split, rms in rms_by_split.items():
        rms = np.asarray(rms, dtype=np.float64)
        if (
            rms.ndim != 2
            or rms.shape[0] == 0
            or rms.shape[1] != weights.shape[1]
            or not np.isfinite(rms).all()
            or np.any(rms < 0)
        ):
            raise ValueError(f"Invalid audio RMS for {split}")
        amplitudes = rms[:, None, :] * weights[None, :, :]
        totals = amplitudes.sum(-1, keepdims=True)
        if np.any(totals <= 0) or not np.isfinite(totals).all():
            raise ValueError(f"Invalid weighted RMS total for {split}")
        adjusted[split] = (amplitudes / totals).mean(0)
    return np.stack(list(adjusted.values())).mean(0), adjusted


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weight_rows(rows, names, weights, **metadata):
    return [
        {**row, **metadata, **dict(zip(names, values, strict=True))}
        for row, values in zip(rows, weights, strict=True)
    ]


def export_adjusted(output_dir, rows, names, weights, records):
    rms_by_split = {
        split: np.array(
            [[row[name] for name in names] for row in records if row["split"] == split]
        )
        for split in SPLIT_COUNTS
    }
    overall, by_split = adjust_weights(weights, rms_by_split)
    write_csv(
        output_dir / "layer_weights_rms_adjusted.csv",
        weight_rows(
            rows, names, overall, num_audio=len(records), aggregation="equal_split_mean"
        ),
    )
    write_csv(
        output_dir / "layer_weights_rms_adjusted_by_split.csv",
        [
            row
            for split, values in by_split.items()
            for row in weight_rows(
                rows, names, values, split=split, num_audio=len(rms_by_split[split])
            )
        ],
    )


def main(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {args.output_dir}")
    set_seed(args.seed)
    queries, samples = None, []
    if not args.weights_only:
        from datasets import load_dataset

        queries = load_dataset(
            "lca0503/INSPIRE", "query", revision=args.dataset_revision
        )
        samples = query_pool(queries)
    embedder = IRISEmbedder.from_pretrained(
        args.model_name,
        revision=args.revision,
        device=args.device,
        dtype=torch.float32 if args.device == "cpu" else "auto",
    )
    rows = conditions()
    weights = extract_weights(embedder, rows)
    names = layer_names(weights.shape[1])
    config = embedder.model.config
    hidden_size = config.text_config.hidden_size
    title = {
        ("qwen2_5_omni_thinker", 2048): "Qwen2.5-Omni-3B",
        ("gemma4", 1536): "Gemma-4-E2B-it",
    }.get((config.model_type, hidden_size), Path(args.model_name).name)
    metadata = {
        "analysis": "instruction_layer_weights",
        "complete": False,
        "model_name": args.model_name,
        "model_revision": embedder.revision,
        "backbone": config.model_type,
        "text_hidden_size": hidden_size,
        "title": args.title or title,
        "layers": names,
        "conditions": rows,
        "prompt": "general",
        "router_device": "cpu",
        "router_dtype": "float32",
        "audio_dtype": str(next(audio_tower(embedder.model).parameters()).dtype),
        "weights_only": args.weights_only,
        "seed": args.seed,
        "dataset_name": "lca0503/INSPIRE",
        "dataset_revision": args.dataset_revision,
        "num_audio": len(samples),
        "split_counts": SPLIT_COUNTS if samples else {},
        "query_fingerprints": {
            split: queries[split]._fingerprint for split in SPLIT_COUNTS
        }
        if samples
        else {},
        "pool": "First row per (split, audio_id); same query audio pool for all instructions and Document:",
        "rms": "Mean over valid frames of channel RMS, before mixture and projection",
        "adjustment": "Normalize weight * RMS per audio; average within each split, then equally across splits",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run(args.output_dir, metadata)
    write_csv(args.output_dir / "layer_weights.csv", weight_rows(rows, names, weights))
    if not args.weights_only:
        with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        records = measure_rms(embedder, queries, samples, names)
        write_csv(args.output_dir / "scale" / "per_audio_rms.csv", records)
        export_adjusted(args.output_dir, rows, names, weights, records)
    metadata["complete"] = True
    write_run(args.output_dir, metadata)
    print(f"Saved {len(rows)} conditions × {len(names)} layers to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
