#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" training/02_train_recordwise.py "$@"
"$PYTHON_BIN" evaluation/03_eval_recordwise.py
"$PYTHON_BIN" evaluation/compare_results.py
