#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
uv run --with-editable "$PROJECT_ROOT" python "$SCRIPT_DIR/run_benchmarks.py" --config "$SCRIPT_DIR/run_benchmarks.jsonc" "$@"
