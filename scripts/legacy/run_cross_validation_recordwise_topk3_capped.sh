#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" training/03_cross_validate_recordwise.py \
  --cache-root outputs/recordwise_topk3_stride32 \
  --output-dir outputs/recordwise_topk3_stride32_cv5_capped8 \
  --folds 5 \
  --inner-folds 4 \
  --imbalance-strategy capped \
  --max-class-weight 8 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --scheduler-patience 8 \
  --patience 12
