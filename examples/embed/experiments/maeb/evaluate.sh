#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 MODEL_NAME OUTPUT_DIR [evaluation options...]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
MODEL_NAME=$1
OUTPUT_DIR=$2
shift 2

python "$REPO_ROOT/src/iris/embed/experiments/maeb/run.py" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
