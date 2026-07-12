#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" scripts/02_train_sampler_only_dpu.py \
  --data outputs/real/spectrogram_cache_max_stride32.npz \
  --epochs 30 --batch-size 64 --lr 0.0005 \
  --output-dir outputs/real/train_max_stride32_4class_dpu "$@"
