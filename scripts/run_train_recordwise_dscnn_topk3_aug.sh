#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT="outputs/recordwise_topk3_stride32"

"$PYTHON_BIN" training/02_train_recordwise.py \
  --train-data "$OUT/train/spectrogram_cache.npz" \
  --val-data "$OUT/val/spectrogram_cache.npz" \
  --test-data "$OUT/test/spectrogram_cache.npz" \
  --output-dir "$OUT" \
  --model radar_dscnn2d \
  --augmentation \
  --imbalance-strategy inverse \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --scheduler-patience 8 \
  --patience 12
