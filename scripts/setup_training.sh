#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SWIFT_DIR=$REPO_ROOT/third_party/ms-swift
PATCH_DIR=$REPO_ROOT/patches/ms-swift
BASE_COMMIT=$(cat "$PATCH_DIR/base_commit.txt")

if [[ ! -d $SWIFT_DIR ]]; then
    git clone https://github.com/modelscope/ms-swift.git "$SWIFT_DIR"
    git -C "$SWIFT_DIR" checkout --detach "$BASE_COMMIT"
fi
if [[ $(git -C "$SWIFT_DIR" rev-parse HEAD) != "$BASE_COMMIT" ]]; then
    echo "Expected ms-swift revision $BASE_COMMIT in $SWIFT_DIR" >&2
    exit 1
fi

if git -C "$SWIFT_DIR" apply --reverse --check "$PATCH_DIR/iris.patch" 2>/dev/null; then
    echo "IRIS training patch is already applied."
else
    if [[ -n $(git -C "$SWIFT_DIR" status --porcelain --untracked-files=no) ]]; then
        echo "The ms-swift checkout has other edits; inspect it before applying the patch." >&2
        exit 1
    fi
    git -C "$SWIFT_DIR" apply --check "$PATCH_DIR/iris.patch"
    git -C "$SWIFT_DIR" apply "$PATCH_DIR/iris.patch"
fi
printf '%s\n' "$BASE_COMMIT" > "$SWIFT_DIR/.iris-patch-applied"

# Run this script only in the fresh training environment described in the README.
python -m pip install -e "${REPO_ROOT}[training]" -e "$SWIFT_DIR"
