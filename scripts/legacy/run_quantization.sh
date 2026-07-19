#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for MODE in calib test; do
  "$PYTHON_BIN" quantization/quantize_vai.py \
    --data outputs/real/spectrogram_cache_max_stride32.npz \
    --checkpoint outputs/real/train_max_stride32_4class_dpu/best_model.pth \
    --quant-mode "$MODE" \
    --output-dir outputs/real/vai_quant_dpu_stride32
done
