#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="outputs/recordwise_topk3_stride32"

"$PYTHON_BIN" evaluation/03_eval_recordwise.py \
  --data "$OUT/test/spectrogram_cache.npz" \
  --train-data "$OUT/train/spectrogram_cache.npz" \
  --val-data "$OUT/val/spectrogram_cache.npz" \
  --checkpoint "$OUT/best_model.pth" \
  --output-dir "$OUT" \
  --batch-size 32
