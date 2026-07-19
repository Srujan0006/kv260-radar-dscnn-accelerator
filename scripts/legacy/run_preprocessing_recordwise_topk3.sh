#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" preprocessing/build_recordwise_spectrogram_cache.py \
  --input ../radar_data/radar_iq.npy \
  --output outputs/recordwise_topk3_stride32 \
  --image-size 100 \
  --stride 32 \
  --range-mode topk_energy_mean \
  --top-k 3 \
  --split grouped_stratified
