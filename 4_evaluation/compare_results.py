#!/usr/bin/env python3
"""Compare window-level and record-wise validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scores(metrics: dict) -> dict[str, float]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-metrics",
        default="outputs/real/train_max_stride32_4class_dpu/test_metrics.json",
    )
    parser.add_argument(
        "--recordwise-metrics", default="outputs/recordwise/test_metrics.json"
    )
    parser.add_argument("--output-dir", default="outputs/recordwise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window = scores(json.loads(Path(args.window_metrics).read_text()))
    recordwise = scores(json.loads(Path(args.recordwise_metrics).read_text()))
    comparison = {
        "window_level_stride32": {
            **window,
            "notes": "Engineering/MVP result; overlapping recording windows may cross splits",
        },
        "record_wise_row_modulo": {
            **recordwise,
            "notes": "Row-disjoint validation result; row 0 train, row 1 val, row 2 test",
        },
    }
    rows = [
        "| Validation type | Accuracy | Macro F1 | Weighted F1 | Notes |",
        "|---|---:|---:|---:|---|",
        (
            "| Window-level stride32 | "
            f"{window['accuracy']:.2%} | {window['macro_f1']:.2%} | "
            f"{window['weighted_f1']:.2%} | Engineering/MVP result |"
        ),
        (
            "| Record-wise split | "
            f"{recordwise['accuracy']:.2%} | {recordwise['macro_f1']:.2%} | "
            f"{recordwise['weighted_f1']:.2%} | Row-disjoint validation result |"
        ),
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
