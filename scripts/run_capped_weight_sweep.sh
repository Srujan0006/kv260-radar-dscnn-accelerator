#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

for cap in 4 6 8; do
  output="outputs/recordwise_topk3_stride32_capped${cap}"
  if [[ -f "$output/best_model.pth" && -f "$output/test_metrics.json" ]]; then
    echo "Keeping completed cap-$cap run: $output"
  else
    bash scripts/run_train_recordwise_dscnn_topk3_aug_capped.sh "$cap"
  fi
done

"$PYTHON_BIN" evaluation/compare_imbalance_strategies.py
