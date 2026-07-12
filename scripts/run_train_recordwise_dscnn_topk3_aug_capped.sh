#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA="outputs/recordwise_topk3_stride32"
CAP="${1:-8}"
if [[ ! "$CAP" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Cap must be a positive number, got: $CAP" >&2
  exit 2
fi
CAP_TAG="${CAP//./p}"
OUT="outputs/recordwise_topk3_stride32_capped${CAP_TAG}"

"$PYTHON_BIN" training/02_train_recordwise.py \
  --train-data "$DATA/train/spectrogram_cache.npz" \
  --val-data "$DATA/val/spectrogram_cache.npz" \
  --test-data "$DATA/test/spectrogram_cache.npz" \
  --output-dir "$OUT" \
  --model radar_dscnn2d \
  --augmentation \
  --imbalance-strategy capped \
  --max-class-weight "$CAP" \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --scheduler-patience 8 \
  --patience 12

"$PYTHON_BIN" evaluation/03_eval_recordwise.py \
  --data "$DATA/test/spectrogram_cache.npz" \
  --train-data "$DATA/train/spectrogram_cache.npz" \
  --val-data "$DATA/val/spectrogram_cache.npz" \
  --checkpoint "$OUT/best_model.pth" \
  --output-dir "$OUT" \
  --batch-size 32
