"""Compare instruction-ablation conditions by split or relevance, with optional CSV."""

import argparse
from pathlib import Path

from iris.embed.experiments.inspire.data import read_run
from iris.embed.experiments.inspire.report import (
    DEFAULT_METRICS,
    get_columns,
    get_headers,
    normalize_metric,
    score_row,
    write_report,
)

IDENTIFIERS = ("Model", "Prompt", "Relevance filter", "Instruction")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Instruction-ablation roots to compare",
    )
    parser.add_argument(
        "--instructions",
        nargs="+",
        help='Exact instruction strings; use "" for the empty control',
    )
    parser.add_argument(
        "--relevances", nargs="+", help="Exact relevance filters; omit for all runs"
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--group_by", choices=("splits", "relevance"), default="splits")
    parser.add_argument("--csv_file", type=Path)
    return parser.parse_args()


def main(args):
    metrics = list(dict.fromkeys(normalize_metric(metric) for metric in args.metrics))
    columns = get_columns(metrics, args.group_by)
    rows = []
    for root in args.run_dirs:
        index = read_run(root)
        if index.get("analysis_type") != "instruction_ablation" or not index.get(
            "complete"
        ):
            raise ValueError(f"Not a completed instruction-ablation run: {root}")
        for relative in index["runs"]:
            directory = root / relative
            metadata = read_run(directory)
            if (
                args.instructions is not None
                and metadata["instruction"] not in args.instructions
            ):
                continue
            if (
                args.relevances is not None
                and metadata["relevance_filter"] not in args.relevances
            ):
                continue
            values = score_row(directory, metrics, columns, args.group_by)
            rows.append(
                [
                    *values[:2],
                    metadata["relevance_filter"] or "all",
                    metadata["instruction"],
                    *values[2:],
                ]
            )
    if not rows:
        raise ValueError(
            "No instruction-ablation conditions matched the requested filters"
        )
    write_report(
        get_headers(columns, args.group_by, IDENTIFIERS),
        rows,
        args.csv_file,
        identifier_count=len(IDENTIFIERS),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
