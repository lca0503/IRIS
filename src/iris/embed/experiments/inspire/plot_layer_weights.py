"""Plot saved instruction-layer weights without loading a checkpoint or audio."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iris.embed.experiments.inspire.data import read_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--include_short", action="store_true")
    parser.add_argument("--font_size", type=float, default=14)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.font_size <= 0 or args.dpi <= 0:
        parser.error("--font_size and --dpi must be positive")
    return args


def load_run(directory):
    metadata = read_run(directory)
    if metadata.get("analysis") != "instruction_layer_weights" or not metadata.get(
        "complete"
    ):
        raise ValueError(f"Expected a completed layer-weight run: {directory}")
    names = metadata["layers"]
    if names != ["pre_encoder"] + [f"encoder_{i}" for i in range(len(names) - 1)]:
        raise ValueError(f"Unexpected layer order in {directory}")
    filenames = {"raw": "layer_weights.csv"}
    if not metadata["weights_only"]:
        filenames["rms_adjusted"] = "layer_weights_rms_adjusted.csv"
    tables = {}
    for variant, filename in filenames.items():
        with (directory / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        conditions = [
            {key: row[key] for key in metadata["conditions"][0]} for row in rows
        ]
        if conditions != metadata["conditions"]:
            raise ValueError(
                f"CSV conditions do not match run.json: {directory / filename}"
            )
        weights = np.array([[float(row[name]) for name in names] for row in rows])
        if (
            weights.shape != (len(conditions), len(names))
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.allclose(weights.sum(1), 1, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"Invalid weights in {directory / filename}")
        if variant == "rms_adjusted" and any(
            int(row["num_audio"]) != metadata["num_audio"]
            or row["aggregation"] != "equal_split_mean"
            for row in rows
        ):
            raise ValueError(
                f"RMS-adjusted sample counts or aggregation differ: {directory}"
            )
        tables[variant] = weights
    return metadata, tables


def color_limits(runs):
    """Share limits across variants/checkpoints of the same backbone and size."""
    limits = {}
    for metadata, tables in runs:
        key = (
            metadata["backbone"],
            metadata["text_hidden_size"],
            len(metadata["layers"]),
        )
        maximum = max(float(weights.max()) for weights in tables.values())
        limits[key] = max(limits.get(key, 0), maximum)
    return limits


def display_label(row, include_short):
    if row["role"] == "document":
        return "Document"
    label = {
        "contain answer": "Find the answer",
        "same speaking style": "Same speaking\nstyle",
        "same environmental sound": "Same environmental\nsound",
    }.get(row["criterion"], row["criterion"].capitalize())
    if include_short:
        label += " (short)" if row["form"] == "short" else " (full)"
    return label


def draw_heatmap(metadata, weights, vmax, font_size=14, include_short=False):
    rows = metadata["conditions"]
    indices = [
        i for i, row in enumerate(rows) if include_short or row["form"] != "short"
    ]
    height = 4 if not include_short else 6
    figure = plt.figure(figsize=(8, height))
    axis = figure.add_axes(
        [0.30, 0.34 if height == 4 else 0.24, 0.68, 0.55 if height == 4 else 0.68]
    )
    mesh = axis.imshow(
        weights[indices], aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax
    )
    axis.set_title(metadata["title"], fontsize=font_size * 1.3, pad=10)
    axis.set_yticks(
        range(len(indices)),
        [display_label(rows[i], include_short) for i in indices],
        fontsize=font_size * 0.9,
    )
    axis.set_xticks(
        range(len(metadata["layers"])),
        ["pre"] + [str(i) for i in range(len(metadata["layers"]) - 1)],
        fontsize=font_size * 0.7,
        rotation=90 if len(metadata["layers"]) > 20 else 0,
    )
    axis.set_xlabel("Audio encoder layer", fontsize=font_size, labelpad=6)
    axis.tick_params(length=0, pad=4)
    for label in axis.get_yticklabels():
        label.set_linespacing(1.0)
    bar_axis = figure.add_axes([0.52, 0.09, 0.44, 0.045 if height == 4 else 0.03])
    colorbar = figure.colorbar(mesh, cax=bar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=font_size * 0.85)
    colorbar.ax.text(
        -0.08,
        0.5,
        "Layer Weights",
        transform=colorbar.ax.transAxes,
        ha="right",
        va="center",
        fontsize=font_size,
    )
    return figure


def main(args):
    directories = list(dict.fromkeys(args.run_dirs))
    runs = [load_run(directory) for directory in directories]
    limits = color_limits(runs)
    for directory, (metadata, tables) in zip(directories, runs, strict=True):
        key = (
            metadata["backbone"],
            metadata["text_hidden_size"],
            len(metadata["layers"]),
        )
        output_dir = directory / "plots"
        output_dir.mkdir(exist_ok=True)
        suffix = "_all_instructions" if args.include_short else ""
        for variant, weights in tables.items():
            figure = draw_heatmap(
                metadata, weights, limits[key], args.font_size, args.include_short
            )
            for extension in ("png", "pdf"):
                figure.savefig(
                    output_dir / f"{variant}{suffix}.{extension}",
                    dpi=args.dpi,
                    facecolor="white",
                )
            plt.close(figure)
        settings = {
            "vmin": 0,
            "vmax": limits[key],
            "scale_group": list(key),
            "include_short": args.include_short,
            "font_size": args.font_size,
            "dpi": args.dpi,
        }
        (output_dir / f"settings{suffix}.json").write_text(
            json.dumps(settings, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved PNG and PDF heatmaps to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
