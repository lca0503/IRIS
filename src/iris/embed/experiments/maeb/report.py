"""Compare completed MAEB summaries and optionally export a CSV."""

import argparse
import csv
import json
import math
from pathlib import Path

from iris.embed.experiments.maeb.tasks import TASK_COLUMNS, TASKS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--csv_file", type=Path)
    return parser.parse_args()


def load_summary(directory):
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if metadata.get("benchmark") != "MAEB" or not metadata.get("complete"):
        raise ValueError(f"MAEB evaluation is incomplete: {directory}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if set(summary) != set(metadata["settings"]["tasks"]):
        raise ValueError(
            f"Summary does not match this run's selected tasks: {directory}"
        )
    for task, score in summary.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
        ):
            raise ValueError(f"Invalid score for {task} in {directory}")
    return metadata["settings"], summary


def main(args):
    columns = [
        (task, label)
        for task, label in zip(TASKS, TASK_COLUMNS, strict=True)
        if task in args.tasks
    ]
    headers = ["Model", "Embedding side", *(label for _, label in columns)]
    rows = []
    for directory in dict.fromkeys(args.run_dirs):
        settings, summary = load_summary(directory)
        rows.append(
            [
                settings["label"],
                settings["embedding_side"],
                *(
                    f"{summary[task]:.6f}" if task in summary else ""
                    for task, _ in columns
                ),
            ]
        )
    print("\t".join(headers))
    for row in rows:
        print("\t".join(row))
    if args.csv_file:
        args.csv_file.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Saved {len(rows)} model rows to {args.csv_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
