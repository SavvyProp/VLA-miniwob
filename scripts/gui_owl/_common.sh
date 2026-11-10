#!/usr/bin/env bash
# Shared launchers; invoke fetch_*, train_*, or eval_* rather than this file.
set -euo pipefail

GUI_OWL_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

gui_owl_help_requested() {
    local argument
    for argument in "$@"; do
        if [[ "$argument" == --help || "$argument" == -h ]]; then
            return 0
        fi
    done
    return 1
}

gui_owl_require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        printf '%s\n' 'uv is required. Install uv, then rerun this script.' >&2
        return 127
    fi
}

gui_owl_fetch() {
    local size="$1"
    shift
    if gui_owl_help_requested "$@"; then
        printf 'Usage: %s\n' "fetch_${size}.sh"
        printf '%s\n' \
            'Fetch the configured immutable checkpoint and processor into the Hugging Face cache.' \
            'Rerunning reuses downloaded files. Set HF_HOME or HF_HUB_CACHE to choose the cache.' \
            'Legacy Transformers cache variables are also honored. No CUDA runtime is needed.'
        return 0
    fi
    if (( $# )); then
        printf '%s\n' 'Fetch takes no arguments; use --help for cache configuration.' >&2
        return 2
    fi
    gui_owl_require_uv
    exec uv run --no-project --python 3.11 --with huggingface-hub==0.36.2 \
        python "$GUI_OWL_REPO_ROOT/scripts/gui_owl/_fetch.py" \
        "$GUI_OWL_REPO_ROOT/configs/gui_owl_${size}_3090.json"
}

gui_owl_train() {
    local size="$1"
    shift
    if gui_owl_help_requested "$@"; then
        printf 'Usage: train_%s.sh [SOURCE [OUTPUT]] [TRAIN_OPTIONS...]\n' "$size"
        printf 'Default source: datasets/seeclick-800-code\nDefault output: runs/gui-owl-%s-code\n' \
            "$size"
        printf 'Configuration: configs/gui_owl_%s_3090.json\n' "$size"
        printf '%s\n' \
            'Examples: --prepare-only; --resume; --resume --max-steps 1266; --history-images 0' \
            'Defaults are relative to the repository. Supplied paths are relative to your current directory.' \
            'Additional options are passed to scriptwob train-gui-owl; --config is fixed by this script.'
        return 0
    fi
    local source="$GUI_OWL_REPO_ROOT/datasets/seeclick-800-code"
    local output="$GUI_OWL_REPO_ROOT/runs/gui-owl-${size}-code"
    if (( $# )) && [[ "$1" != -* ]]; then source="$1"; shift; fi
    if (( $# )) && [[ "$1" != -* ]]; then output="$1"; shift; fi
    local argument
    for argument in "$@"; do
        if [[ "$argument" == --config || "$argument" == --config=* ]]; then
            printf 'Edit configs/gui_owl_%s_3090.json, or use train-gui-owl directly for a custom config.\n' \
                "$size" >&2
            return 2
        fi
    done
    gui_owl_require_uv
    exec uv run --project "$GUI_OWL_REPO_ROOT/integrations/gui_owl" --extra cuda --frozen \
        scriptwob train-gui-owl "$source" "$output" \
        --config "$GUI_OWL_REPO_ROOT/configs/gui_owl_${size}_3090.json" "$@"
}

gui_owl_eval() {
    local size="$1"
    shift
    if gui_owl_help_requested "$@"; then
        printf 'Usage: eval_%s.sh [CHECKPOINT [MANIFEST [OUTPUT_PREFIX]]] [EVAL_OPTIONS...]\n' "$size"
        printf 'Default checkpoint and output prefix: runs/gui-owl-%s-code\n' "$size"
        printf '%s\n' \
            'Default manifest: configs/suite_800.json' \
            'Evaluates dev then test, writing OUTPUT_PREFIX-dev and OUTPUT_PREFIX-test.' \
            'Examples: --fresh; --max-total-steps 20000; --history-images 0' \
            'Defaults are relative to the repository. Supplied paths are relative to your current directory.' \
            'Additional options are passed to both splits. The prediction budget applies to each split.' \
            'Use evaluate-gui-owl directly to run only one split.'
        return 0
    fi
    local checkpoint="$GUI_OWL_REPO_ROOT/runs/gui-owl-${size}-code"
    local manifest="$GUI_OWL_REPO_ROOT/configs/suite_800.json"
    local output="$GUI_OWL_REPO_ROOT/runs/gui-owl-${size}-code"
    if (( $# )) && [[ "$1" != -* ]]; then checkpoint="$1"; shift; fi
    if (( $# )) && [[ "$1" != -* ]]; then manifest="$1"; shift; fi
    if (( $# )) && [[ "$1" != -* ]]; then output="$1"; shift; fi
    local argument
    for argument in "$@"; do
        if [[ "$argument" == --split || "$argument" == --split=* ]]; then
            printf '%s\n' 'This script evaluates both splits; use evaluate-gui-owl directly for --split.' >&2
            return 2
        fi
    done
    gui_owl_require_uv
    local split
    for split in dev test; do
        printf 'Evaluating GUI-Owl %s: %s → %s-%s\n' "$size" "$split" "$output" "$split"
        uv run --project "$GUI_OWL_REPO_ROOT/integrations/gui_owl" --extra cuda --frozen \
            scriptwob evaluate-gui-owl "$checkpoint" "$manifest" "${output}-${split}" \
            --split "$split" --max-total-steps 10000 "$@" || return "$?"
    done
}
