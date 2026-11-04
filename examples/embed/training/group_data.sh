#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 TRAIN_JSONL GROUPED_TRAIN_JSONL [grouping options...]" >&2
    exit 2
fi
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
INPUT_FILE=$1
OUTPUT_FILE=$2
shift 2
python "$REPO_ROOT/src/iris/embed/training/group_embedding_data.py" \
    --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" "$@"
