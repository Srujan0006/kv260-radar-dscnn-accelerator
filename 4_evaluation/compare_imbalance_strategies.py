#!/usr/bin/env python3
"""Compare strong weights, sampler-only, and capped-weight record-wise runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", default="outputs/recordwise_topk3_stride32")
    parser.add_argument(
        "--sampler", default="outputs/recordwise_topk3_stride32_sampler_only"
    )
    parser.add_argument("--cap4", default="outputs/recordwise_topk3_stride32_capped4")
    parser.add_argument("--cap6", default="outputs/recordwise_topk3_stride32_capped6")
    parser.add_argument(
        "--cap8",
        "--capped",
        dest="cap8",
        default="outputs/recordwise_topk3_stride32_capped8",
    )
    parser.add_argument("--output-dir", default="outputs/imbalance_strategy_comparison")
    return parser.parse_args()


def load_result(name: str, path: Path) -> dict:
    metrics_path = path / "test_metrics.json"
    train_path = path / "train_results.json"
    if not metrics_path.exists():
        return {"name": name, "status": "not_run", "path": str(path)}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training = (
        json.loads(train_path.read_text(encoding="utf-8"))
        if train_path.exists()
        else {}
    )
    return {
        "name": name,
        "status": "complete",
        "path": str(path),
        "imbalance_strategy": training.get("imbalance_strategy", name),
        "window_accuracy": metrics["test_accuracy"],
        "window_macro_f1": metrics["macro_f1"],
        "window_weighted_f1": metrics["weighted_f1"],
        "predicted_distribution": metrics.get("predicted_distribution"),
        "recording_accuracy": metrics.get("recording_level", {}).get("accuracy"),
        "recording_macro_f1": metrics.get("recording_level", {}).get("macro_f1"),
        "recording_weighted_f1": metrics.get("recording_level", {}).get("weighted_f1"),
        "recording_row_overlap": metrics["recording_row_overlap"],
    }


def percent(value: float | None) -> str:
    return "TBD" if value is None else f"{value:.2%}"


def main() -> None:
    args = parse_args()
    results = [
        load_result("Strong inverse weights", Path(args.strong)),
        load_result("Sampler only", Path(args.sampler)),
        load_result("Capped weights (4)", Path(args.cap4)),
        load_result("Capped weights (6)", Path(args.cap6)),
        load_result("Capped weights (8)", Path(args.cap8)),
    ]
    rows = [
        "| Strategy | Window accuracy | Window macro F1 | Window weighted F1 | Recording accuracy | Recording macro F1 | Row overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if result["status"] == "complete":
            rows.append(
                f"| {result['name']} | {percent(result['window_accuracy'])} | "
                f"{percent(result['window_macro_f1'])} | "
                f"{percent(result['window_weighted_f1'])} | "
                f"{percent(result['recording_accuracy'])} | "
                f"{percent(result['recording_macro_f1'])} | "
                f"{result['recording_row_overlap']} |"
            )
        else:
            rows.append(f"| {result['name']} | TBD | TBD | TBD | TBD | TBD | TBD |")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
