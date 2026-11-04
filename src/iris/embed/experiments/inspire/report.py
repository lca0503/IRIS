"""Compare completed INSPIRE runs by split or relevance; optionally export CSV."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean

from iris.embed.experiments.inspire.data import REPORT_SPLITS, read_run

DEFAULT_METRICS = (
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
)
RELEVANCE_GROUPS = {
    "DailyTalk": ("continuation",),
    "VCTK": ("same speaker",),
    "Expresso": (
        "same speaker",
        "same speaking style",
        "same speaker and same speaking style",
        "same speaker and specific speaking style",
    ),
    "Synthetic": (
        "contain answer",
        "same speaker",
        "same speaking style",
        "same environmental sound",
        "same speaker and same speaking style",
        "same speaker and specific speaking style",
        "same speaker and same environmental sound",
        "same speaker and specific environmental sound",
        "same speaking style and same environmental sound",
        "same speaking style and specific environmental sound",
        "specific speaking style and same environmental sound",
        "same speaker and same speaking style and same environmental sound",
        "same speaker and specific speaking style and same environmental sound",
        "same speaker and same speaking style and specific environmental sound",
        "same speaker and specific speaking style and specific environmental sound",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories to compare, in row order",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Recall@k, R@k, or NDCG@k",
    )
    parser.add_argument("--group_by", choices=("splits", "relevance"), default="splits")
    parser.add_argument("--csv_file", type=Path)
    return parser.parse_args()


def normalize_metric(metric):
    name, separator, k = metric.lower().partition("@")
    if (
        not separator
        or not k.isdigit()
        or int(k) <= 0
        or name not in {"r", "recall", "ndcg"}
    ):
        raise ValueError(
            f"Invalid metric {metric!r}; use Recall@k or NDCG@k with k > 0"
        )
    return f"{'recall' if name in {'r', 'recall'} else 'ndcg'}@{int(k)}"


def normalize_relevance(relevance):
    """Group named target styles/sounds under specific style/sound, as in INSPIRE."""
    if not relevance:
        return relevance

    def replace(match):
        if match.group(1) == "same":
            return match.group(0)
        return f"specific {match.group(2)}"

    relevance = re.sub(
        r"([\w_]+)\s+(speaking style|environmental sound)", replace, relevance
    )
    return re.sub(r"\s+", " ", relevance.replace(" and and ", " and ")).strip()


def relevance_label(relevance):
    for old, new in (
        ("same speaker", "same spk."),
        ("specific speaking style", "spec sty."),
        ("same speaking style", "same sty."),
        ("specific environmental sound", "spec env."),
        ("same environmental sound", "same env."),
        (" and ", " "),
    ):
        relevance = relevance.replace(old, new)
    return relevance


def get_columns(metrics, group_by):
    if group_by == "splits":
        return [(split, "", metric) for split in REPORT_SPLITS for metric in metrics]
    return [
        (split, relevance, metric)
        for split in REPORT_SPLITS
        for relevance in RELEVANCE_GROUPS[split]
        for metric in metrics
    ]


def load_scores(run_dir, metadata):
    if not metadata.get("scoring", {}).get("complete"):
        raise ValueError(f"Scoring is incomplete: {run_dir}")
    scores = {}
    for split in metadata["splits"]:
        path = run_dir / "scores" / split / "score.jsonl"
        with path.open(encoding="utf-8") as file:
            entries = [json.loads(line) for line in file if line.strip()]
        expected_ids = [
            f"{index:05d}" for index in range(metadata["dataset"][split]["num_queries"])
        ]
        if not entries or [entry["id"] for entry in entries] != expected_ids:
            raise ValueError(f"Incomplete or mismatched score rows: {path}")
        scores[split] = entries
    return scores


def score_row(run_dir, metrics, columns, group_by):
    metadata = read_run(run_dir)
    grouped = {}
    for split, entries in load_scores(run_dir, metadata).items():
        for entry in entries:
            for metric in metrics:
                if metric not in entry:
                    raise ValueError(
                        f"{run_dir} lacks {metric}; rerun score.py with that k value"
                    )
                if not math.isfinite(entry[metric]) or not 0 <= entry[metric] <= 1:
                    raise ValueError(f"Invalid {metric} in {run_dir}/{split}")
            relevance = (
                normalize_relevance(entry["relevance"])
                if group_by == "relevance"
                else ""
            )
            if group_by == "relevance" and relevance not in RELEVANCE_GROUPS[split]:
                raise ValueError(f"Unknown relevance in {split}: {relevance!r}")
            grouped.setdefault((split, relevance), []).append(entry)
    values = [
        fmean(entry[metric] for entry in grouped[(split, relevance)])
        if (split, relevance) in grouped
        else None
        for split, relevance, metric in columns
    ]
    return [metadata["label"], metadata["prompt"], *values]


def get_headers(columns, group_by, identifiers=("Model", "Prompt")):
    blanks = [""] * len(identifiers)
    headers = [[*identifiers, *(split for split, _, _ in columns)]]
    if group_by == "relevance":
        headers.append([*blanks, *(relevance_label(label) for _, label, _ in columns)])
    labels = []
    for _, _, metric in columns:
        name, k = metric.split("@")
        labels.append(f"{'Recall' if name == 'recall' else 'NDCG'}@{k}")
    headers.append([*blanks, *labels])
    return headers


def write_report(headers, rows, csv_file=None, identifier_count=2):
    formatted = [
        [
            *row[:identifier_count],
            *(
                "" if value is None else f"{value:.6f}"
                for value in row[identifier_count:]
            ),
        ]
        for row in rows
    ]
    for row in (*headers, *formatted):
        print("\t".join(row))
    if csv_file is not None:
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with csv_file.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerows((*headers, *formatted))
        print(f"Results saved to: {csv_file}")


def main(args):
    metrics = list(dict.fromkeys(normalize_metric(metric) for metric in args.metrics))
    columns = get_columns(metrics, args.group_by)
    rows = [
        score_row(directory, metrics, columns, args.group_by)
        for directory in args.run_dirs
    ]
    write_report(get_headers(columns, args.group_by), rows, args.csv_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
