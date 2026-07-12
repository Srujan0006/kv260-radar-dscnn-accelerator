#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/01_make_spectrogram_cache.py \
  --input data/real/radar_iq_windows_stride32_4class.npz \
  --output outputs/real/spectrogram_cache_max_stride32.npz \
  --image-size 100 \
  --range-mode max

"$PYTHON_BIN" preprocessing/01_make_recordwise_windows.py \
  --input ../radar_data/radar_iq.npy \
  --output-dir outputs/recordwise \
  --stride 32 \
  --range-mode max \
  --split-strategy row_modulo
