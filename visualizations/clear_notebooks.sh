#!/usr/bin/env bash
# Clear all outputs from Jupyter notebooks in this directory and subdirectories.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

find "$SCRIPT_DIR" -name "*.ipynb" | while read -r nb; do
    echo "Clearing: $nb"
    jq '(.cells[].outputs = []) | (.cells[].execution_count = null)' "$nb" > "$nb.tmp" && mv "$nb.tmp" "$nb"
done
