#!/usr/bin/env bash
# Install and check the isolated TinyVLA environment; download no model weights.
set -euo pipefail

backend="${1:-cuda}"
if [[ "$backend" != "cpu" && "$backend" != "cuda" ]] || [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/setup_tinyvla.sh [cuda|cpu]" >&2
    exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv sync --project "$repo_root/integrations/tinyvla" --frozen --extra "$backend"
check_args=()
if [[ "$backend" == "cuda" ]]; then
    check_args+=(--require-cuda)
fi
uv run --project "$repo_root/integrations/tinyvla" --frozen --extra "$backend" \
    python "$repo_root/scripts/check_tinyvla.py" "${check_args[@]}"
