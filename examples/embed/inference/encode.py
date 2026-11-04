"""Encode query or document recordings with an IRIS-Embed checkpoint."""

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
    parser.add_argument("--audio", nargs="+", required=True, type=Path)
    parser.add_argument("--role", choices=("query", "document"), required=True)
    parser.add_argument("--instruction", help="Required for query; omit for document")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output_file", type=Path, required=True, help="Output .npy")
    args = parser.parse_args()
    if args.role == "query" and not (args.instruction and args.instruction.strip()):
        parser.error("Query encoding requires a nonempty --instruction")
    if args.role == "document" and args.instruction is not None:
        parser.error("Document encoding does not take --instruction")
    if args.output_file.suffix != ".npy":
        parser.error("--output_file must end in .npy")
    return args


def main(args):
    if args.output_file.exists():
        raise FileExistsError(f"Output already exists: {args.output_file}")
    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Audio file does not exist: {missing[0]}")
    model = IRISEmbedder.from_pretrained(
        args.model_name, revision=args.revision, device=args.device
    )
    if args.role == "query":
        embeddings = model.encode_queries(args.audio, args.instruction)
    else:
        embeddings = model.encode_documents(args.audio)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_file, embeddings)
    print(f"Saved {embeddings.shape} normalized embeddings to {args.output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
