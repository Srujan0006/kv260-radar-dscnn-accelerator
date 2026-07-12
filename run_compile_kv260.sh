#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec bash quantization/compile_kv260.sh \
  outputs/real/vai_quant_dpu_stride32/Radar2DCNN_int.xmodel \
  outputs/real/compiled_kv260_dpu "$@"
