"""Evaluate a Hugging Face IRIS checkpoint on the 15 MAEB tasks."""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import mteb
from mteb.models import ModelMeta

from iris.embed.experiments.maeb.adapter import (
    EMBEDDING_SIDES,
    MAX_AUDIO_SECONDS,
    MAEBEncoder,
)
from iris.embed.experiments.maeb.tasks import QUERY_DOCUMENT_TASKS, TASK_INSTRUCTIONS, TASKS
from iris.embed.inference import IRISEmbedder

MTEB_VERSION = "2.18.0"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name", required=True, help="HF ID or local exported IRIS checkpoint"
    )
    parser.add_argument("--revision", help="Model revision")
    parser.add_argument(
        "--label", help="Display name in reports; defaults to the checkpoint name"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--embedding_side",
        choices=EMBEDDING_SIDES,
        default="query",
        help="Side for classification/clustering/pair tasks; retrieval/reranking always use query -> document",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed tasks from this run with identical settings",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    return args


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_run(args):
    """Keep each run isolated; resume only the same model, prompt and task set."""
    if version("mteb") != MTEB_VERSION:
        raise ValueError(f"This experiment requires mteb=={MTEB_VERSION}")
    settings = {
        "model_name": args.model_name,
        "requested_revision": args.revision,
        "label": args.label or Path(args.model_name).name,
        "tasks": [task for task in TASKS if task in args.tasks],
        "embedding_side": args.embedding_side,
        "batch_size": args.batch_size,
        "prompt": "general",
        "normalization": "l2",
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "mteb_version": MTEB_VERSION,
    }
    prior = None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Use a new output directory or --resume: {args.output_dir}"
            )
        prior = json.loads((args.output_dir / "run.json").read_text(encoding="utf-8"))
        if prior.get("benchmark") != "MAEB" or prior["settings"] != settings:
            raise ValueError(
                "Resume requires identical model, revision, prompt, tasks, batch size and label"
            )
    elif args.resume:
        raise FileNotFoundError(f"No MAEB run to resume: {args.output_dir}")
    return settings, prior


def model_metadata(settings, embedder):
    uses_instructions = settings["embedding_side"] != "document" or bool(
        QUERY_DOCUMENT_TASKS.intersection(settings["tasks"])
    )
    return ModelMeta(
        loader=None,
        name=f"iris/{Path(settings['model_name']).name}-{settings['embedding_side']}",
        revision=embedder.revision or "local",
        release_date=None,
        languages=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=embedder.model.config.text_config.hidden_size,
        license=None,
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        framework=["PyTorch", "Transformers"],
        similarity_fn_name="cosine",
        use_instructions=uses_instructions,
        training_datasets=None,
        modalities=["audio"],
    )


def collect_scores(results, tasks):
    summary = {}
    for result in results:
        if result.task_name in summary:
            raise ValueError(f"Duplicate MAEB result: {result.task_name}")
        score = float(result.get_score())
        if not math.isfinite(score):
            raise ValueError(f"Non-finite MAEB score: {result.task_name}")
        summary[result.task_name] = score
    if set(summary) != set(tasks):
        raise ValueError("MAEB results do not cover exactly the selected tasks")
    return {task: summary[task] for task in tasks}


def main(args):
    settings, prior = prepare_run(args)
    tasks = list(mteb.get_tasks(tasks=settings["tasks"]))
    if [task.metadata.name for task in tasks] != settings["tasks"]:
        raise ValueError("MTEB task selection differs from the requested MAEB tasks")
    revision = prior["model_revision"] if prior else args.revision
    embedder = IRISEmbedder.from_pretrained(
        args.model_name, revision=revision, device=args.device
    )
    encoder = MAEBEncoder(embedder, args.embedding_side)
    encoder.mteb_model_meta = model_metadata(settings, embedder)
    metadata = {
        "benchmark": "MAEB",
        "complete": False,
        "settings": settings,
        "model_revision": embedder.revision,
        "versions": {
            package: version(package)
            for package in (
                "mteb",
                "datasets",
                "transformers",
                "torch",
                "torchcodec",
                "scikit-learn",
                "numpy",
                "scipy",
            )
        },
        "pooling": "last non-padding LLM token",
        "retrieval_reranking_sides": {"query": "query", "corpus": "document"},
        "instructions": {task: TASK_INSTRUCTIONS[task] for task in settings["tasks"]},
        "task_metadata": {
            task.metadata.name: {
                "dataset": task.metadata.dataset,
                "eval_splits": task.metadata.eval_splits,
                "main_score": task.metadata.main_score,
                "seed": task.seed,
            }
            for task in tasks
        },
    }
    if prior and (
        prior["model_revision"] != metadata["model_revision"]
        or prior["versions"] != metadata["versions"]
    ):
        raise ValueError(
            "Resume requires the same resolved checkpoint and package versions"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run.json", metadata)
    print(
        f"Evaluating {len(tasks)} MAEB tasks; default side: {args.embedding_side}",
        flush=True,
    )
    results = mteb.evaluate(
        encoder,
        tasks,
        cache=mteb.ResultCache(cache_path=args.output_dir / "mteb"),
        encode_kwargs={"batch_size": args.batch_size},
        overwrite_strategy="only-missing",
        co2_tracker=False,
        raise_error=True,
    )
    summary = collect_scores(results, settings["tasks"])
    write_json(args.output_dir / "summary.json", summary)
    metadata["complete"] = True
    write_json(args.output_dir / "run.json", metadata)
    for task, score in summary.items():
        print(f"{task}: {score:.6f}")
    print(f"Saved MAEB results to {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
