#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 BASE_RUN_DIR OUTPUT_DIR [analysis options...]" >&2
    exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
BASE_RUN_DIR=$1
OUTPUT_DIR=$2
shift 2

python "$REPO_ROOT/src/iris/embed/experiments/inspire/instruction_ablation.py" \
    --base_run_dir "$BASE_RUN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --splits Synthetic \
    --relevances \
        "same speaker and same speaking style" \
        "same speaker and same environmental sound" \
        "same speaking style and same environmental sound" \
        "same speaker and same speaking style and same environmental sound" \
    --instructions \
        "Find the answer." \
        "Continuation." \
        "Same speaker." \
        "Same speaking style." \
        "Same environmental sound." \
        "Same speaker and same environmental sound." \
        "Same speaker and same speaking style." \
        "Same speaker and same speaking style and same environmental sound." \
        "Same speaking style and same environmental sound." \
        "" \
    "$@"
