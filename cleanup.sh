#!/bin/bash
set -euo pipefail
#
# Idempotent cleanup: delete demo bucket + local temp files.
#

extract_user() {
    # Handle both TTY ("  user: rajatarya") and non-TTY ("user=rajatarya")
    # forms; strip ANSI color codes first.
    hf auth whoami 2>&1 \
        | sed -E 's/\x1b\[[0-9;]*m//g' \
        | sed -nE 's/.*user[ :=]+([A-Za-z0-9_-]+).*/\1/p' \
        | head -1
}

ARG="${1:-}"
if [ -n "$ARG" ]; then
    BUCKET="$ARG"
else
    USER_NAME="$(extract_user)"
    if [ -z "$USER_NAME" ]; then
        echo "Could not parse username from 'hf auth whoami'. Pass bucket name as arg." >&2
        exit 1
    fi
    BUCKET="${USER_NAME}/nvidia-simready"
fi

echo ">>> Deleting bucket $BUCKET (if exists)"
hf buckets delete "$BUCKET" --yes 2>/dev/null || echo "    (bucket did not exist)"

echo ">>> Removing local temp files"
rm -f /tmp/original.csv /tmp/annotated.csv /tmp/nvidia-simready-summary.json

echo ">>> Removing Python cache dirs under $(dirname "$0")"
find "$(dirname "$0")" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

echo "Done."
