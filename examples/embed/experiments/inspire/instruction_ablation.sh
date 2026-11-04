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
    --instructions \
        "" \
        "Continuation." \
        "Find speech that continues the conversation in the query." \
        "Same speaker." \
        "Find speech from the same speaker as the query." \
        "Same speaking style." \
        "Find speech with the same speaking style as the query." \
        "Same environmental sound." \
        "Find speech with the same environmental sound as the query." \
        "Find the answer." \
        "Find speech that contains the answer to the question in the query." \
    "$@"
