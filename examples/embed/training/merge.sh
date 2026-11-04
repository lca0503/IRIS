#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash $0 ADAPTER_DIR OUTPUT_DIR [AUDIO_FILE]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SWIFT_DIR=$REPO_ROOT/third_party/ms-swift
if [[ ! -f $SWIFT_DIR/.iris-patch-applied ]]; then
    echo "Run bash scripts/setup_training.sh first." >&2
    exit 2
fi
if [[ -e $2 ]]; then
    echo "Output already exists: $2" >&2
    exit 2
fi
export PYTHONPATH="$SWIFT_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m swift.cli.main export \
    --adapters "$1" --load_args true --merge_lora true \
    --output_dir "$2" --use_hf true
python "$REPO_ROOT/src/iris/embed/training/prepare_hf_checkpoint.py" --checkpoint_dir "$2"

VALIDATE_ARGS=(--checkpoint_dir "$2")
if [[ $# -eq 3 ]]; then
    VALIDATE_ARGS+=(--audio_file "$3")
fi
python "$REPO_ROOT/src/iris/embed/training/validate_hf_checkpoint.py" "${VALIDATE_ARGS[@]}"
