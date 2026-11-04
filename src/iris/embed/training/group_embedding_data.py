"""Group embedding rows by spoken query and interleave source blocks."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_groups(input_file):
    groups_by_source = {}
    row_count = 0
    byte_count = 0

    with input_file.open("rb") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"{input_file}:{line_number} must end with a newline before reordering"
                )
            row = json.loads(line)
            source = row.get("source")
            spoken_query = row.get("spoken_query")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{input_file}:{line_number} lacks a valid source")
            if not isinstance(spoken_query, str) or not spoken_query:
                raise ValueError(
                    f"{input_file}:{line_number} lacks a valid spoken_query"
                )
            source_groups = groups_by_source.setdefault(source, {})
            source_groups.setdefault(spoken_query, []).append(line)
            row_count += 1
            byte_count += len(line)

    return groups_by_source, row_count, byte_count


def make_source_blocks(groups_by_source, block_size, rng):
    blocks_by_source = {}
    for source, spoken_query_groups in groups_by_source.items():
        groups = list(spoken_query_groups.values())
        rng.shuffle(groups)

        blocks = []
        block = []
        for group in groups:
            if block and len(block) + len(group) > block_size:
                blocks.append(block)
                block = []
            block.extend(group)
        if block:
            blocks.append(block)

        blocks_by_source[source] = blocks
    return blocks_by_source


def can_finish_without_neighbors(blocks_by_source, selected_source):
    remaining = {
        source: len(blocks) - (source == selected_source)
        for source, blocks in blocks_by_source.items()
    }
    total = sum(remaining.values())
    return all(
        count <= total - count + (source != selected_source)
        for source, count in remaining.items()
    )


def interleave_source_blocks(blocks_by_source, rng):
    total = sum(map(len, blocks_by_source.values()))
    largest = max(map(len, blocks_by_source.values()), default=0)
    if largest > total - largest + 1:
        raise ValueError("Source blocks cannot be arranged without equal neighbors")

    ordered_blocks = []
    previous_source = None
    while blocks_by_source:
        candidates = [
            source
            for source in blocks_by_source
            if source != previous_source
            and can_finish_without_neighbors(blocks_by_source, source)
        ]
        if not candidates:
            raise RuntimeError("Could not select a non-neighboring source block")

        source = rng.choices(
            candidates,
            weights=[len(blocks_by_source[name]) for name in candidates],
            k=1,
        )[0]
        ordered_blocks.append((source, blocks_by_source[source].pop()))
        previous_source = source
        if not blocks_by_source[source]:
            del blocks_by_source[source]

    return ordered_blocks


def main(args):
    if args.block_size <= 0:
        raise ValueError("block_size must be positive")
    if args.input_file.resolve() == args.output_file.resolve():
        raise ValueError("input_file and output_file must be different")
    if args.output_file.exists():
        raise FileExistsError(f"Output already exists: {args.output_file}")

    temporary_file = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    if temporary_file.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_file}")

    rng = random.Random(args.seed)
    groups_by_source, row_count, byte_count = load_groups(args.input_file)
    if not groups_by_source:
        raise ValueError(f"Input contains no rows: {args.input_file}")
    group_count = sum(map(len, groups_by_source.values()))
    blocks_by_source = make_source_blocks(groups_by_source, args.block_size, rng)
    ordered_blocks = interleave_source_blocks(blocks_by_source, rng)
    block_sizes = [len(block) for _, block in ordered_blocks]
    if sum(block_sizes) != row_count:
        raise ValueError("Grouped output row count does not match the input")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with temporary_file.open("wb") as file:
        for _, block in ordered_blocks:
            file.writelines(block)

    if temporary_file.stat().st_size != byte_count:
        raise ValueError("Grouped output size does not match the input size")
    temporary_file.replace(args.output_file)

    print(
        f"Wrote {row_count} unchanged rows from {len(groups_by_source)} sources "
        f"and {group_count} spoken-query groups to {args.output_file}"
    )
    print(
        f"Source blocks: {len(block_sizes)}; size: "
        f"{min(block_sizes)}-{max(block_sizes)}; target: {args.block_size}; "
        f"seed: {args.seed}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
