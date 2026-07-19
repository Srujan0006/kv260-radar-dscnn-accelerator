#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" quantization/inspect_xmodel.py \
  --xmodel outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel "$@"
