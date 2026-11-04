#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash $0 CONFIG TRAIN_JSONL VAL_JSONL OUTPUT_DIR [swift options...]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
CONFIG=$1
export IRIS_TRAIN_FILE=$2
export IRIS_VAL_FILE=$3
OUTPUT_DIR=$4
shift 4

for file in "$CONFIG" "$IRIS_TRAIN_FILE" "$IRIS_VAL_FILE"; do
    if [[ ! -f $file ]]; then
        echo "File does not exist: $file" >&2
        exit 2
    fi
done

# Use only the isolated, patched training framework.
SWIFT_DIR=$REPO_ROOT/third_party/ms-swift
if [[ ! -f $SWIFT_DIR/.iris-patch-applied ]]; then
    echo "Run bash scripts/setup_training.sh first." >&2
    exit 2
fi
export PYTHONPATH="$SWIFT_DIR:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export INFONCE_TEMPERATURE=0.03
export INFONCE_MASK_FAKE_NEGATIVE=True
export INFONCE_USE_BATCH=True
export INFONCE_INCLUDE_QQ=False
export INFONCE_INCLUDE_DD=False
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m swift.cli.main sft "$CONFIG" \
    --dataset iris:train \
    --val_dataset iris:val \
    --custom_register_path "$REPO_ROOT/src/iris/embed/training/register_embedding_data.py" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
