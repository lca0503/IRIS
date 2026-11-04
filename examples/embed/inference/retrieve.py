"""Rank document recordings for one instructed query with an IRIS-Embed checkpoint."""

import argparse
from pathlib import Path

import numpy as np

from iris.embed.inference import IRISEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help=(
            "lca0503/IRIS-Embed-Qwen2.5-Omni-3B\n"
            "lca0503/IRIS-Embed-gemma-4-E2B-it\n"
            "or a local checkpoint directory"
        ),
    )
    parser.add_argument("--revision", help="Checkpoint commit for reproducibility")
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--documents", type=Path, nargs="+", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top_k must be positive")
    if not args.instruction.strip():
        parser.error("--instruction must be nonempty")
    return args


def main(args):
    missing = [path for path in [args.query, *args.documents] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio file does not exist: {missing[0]}")
    model = IRISEmbedder.from_pretrained(
        args.model_name, revision=args.revision, device=args.device
    )
    query = model.encode_queries(args.query, args.instruction)
    documents = model.encode_documents(args.documents)
    scores = (query @ documents.T)[0]
    ranked = np.argsort(-scores, kind="stable")[: args.top_k]
    for rank, index in enumerate(ranked, start=1):
        print(f"{rank}\t{scores[index]:.6f}\t{args.documents[index]}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
