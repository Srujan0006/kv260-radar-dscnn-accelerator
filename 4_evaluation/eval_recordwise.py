#!/usr/bin/env python3
"""Evaluate a record-wise checkpoint on the untouched recording-row test cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import CachedSpectrogramDataset, load_spectrogram_cache
from radar2dcnn.evaluation import (
    RECORDING_AGGREGATION_NAMES,
    aggregate_recording_predictions,
    detailed_classification_metrics,
)
from radar2dcnn.labels import CLASS_NAMES
from radar2dcnn.metrics import save_confusion_matrix
from radar2dcnn.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/recordwise/test/spectrogram_cache.npz")
    parser.add_argument("--train-data", default="outputs/recordwise/train/spectrogram_cache.npz")
    parser.add_argument("--val-data", default="outputs/recordwise/val/spectrogram_cache.npz")
    parser.add_argument("--checkpoint", default="outputs/recordwise/best_model.pth")
    parser.add_argument("--output-dir", default="outputs/recordwise")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--top-confidence-fraction",
        type=float,
        default=0.30,
        help="Fraction of most-confident windows used by top-confidence averaging",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Defaults to split_manifest.json beside the train/val/test cache folders",
    )
    return parser.parse_args()


def rows_in(path: str | Path) -> set[int]:
    with np.load(path, allow_pickle=False) as data:
        if "row_indices" not in data.files:
            raise KeyError(f"{path} lacks row_indices metadata")
        return set(data["row_indices"].astype(int).tolist())


def main() -> None:
    args = parse_args()
    train_rows = rows_in(args.train_data)
    val_rows = rows_in(args.val_data)
    test_rows = rows_in(args.data)
    overlap = (train_rows & val_rows) | (train_rows & test_rows) | (val_rows & test_rows)
    if overlap:
        raise RuntimeError(f"Recording-row leakage detected: {sorted(overlap)}")

    arrays = load_spectrogram_cache(args.data)
    with np.load(args.data, allow_pickle=False) as cache:
        if "row_indices" not in cache.files:
            raise KeyError(f"{args.data} lacks row_indices metadata")
        row_indices = cache["row_indices"].astype(np.int64)
    loader = DataLoader(
        CachedSpectrogramDataset(arrays), batch_size=args.batch_size, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, map_location=device).to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    all_logits: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Record-wise test"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(float(criterion(logits, y).item()))
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_logits.append(logits.cpu().numpy())

    y_true_array = np.asarray(y_true, dtype=np.int64)
    y_pred_array = np.asarray(y_pred, dtype=np.int64)
    logits_array = np.concatenate(all_logits, axis=0)
    window_metrics = detailed_classification_metrics(
        y_true_array, y_pred_array, CLASS_NAMES
    )
    recording_ids, recording_true, recording_predictions = (
        aggregate_recording_predictions(
            logits_array,
            y_true_array,
            row_indices,
            top_fraction=args.top_confidence_fraction,
        )
    )
    aggregation_metrics = {
        method: {
            "aggregation": RECORDING_AGGREGATION_NAMES[method],
            "source_rows": len(recording_ids),
            **detailed_classification_metrics(
                recording_true, recording_pred, CLASS_NAMES
            ),
        }
        for method, recording_pred in recording_predictions.items()
    }
    aggregation_metrics["top_confidence_mean"]["aggregation"] = (
        f"Top-{args.top_confidence_fraction:.0%} confidence average"
    )
    recording_metrics = aggregation_metrics["mean_logits"]
    manifest_path = (
        Path(args.split_manifest)
        if args.split_manifest
        else Path(args.data).resolve().parents[1] / "split_manifest.json"
    )
    split_strategy = "recording_rows"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            split_strategy = json.load(handle).get("strategy", split_strategy)
    output = {
        "validation_level": "window predictions from held-out recording rows",
        "split_strategy": split_strategy,
        "recording_row_overlap": 0,
        "test_windows": len(y_true),
        "test_source_rows": len(test_rows),
        "test_distribution": window_metrics["true_distribution"],
        "predicted_distribution": window_metrics["predicted_distribution"],
        "test_loss": float(np.mean(losses)),
        "test_accuracy": window_metrics["accuracy"],
        "macro_precision": window_metrics["macro_precision"],
        "macro_recall": window_metrics["macro_recall"],
        "macro_f1": window_metrics["macro_f1"],
        "weighted_precision": window_metrics["weighted_precision"],
        "weighted_recall": window_metrics["weighted_recall"],
        "weighted_f1": window_metrics["weighted_f1"],
        "per_class": window_metrics["per_class"],
        "classification_report": window_metrics["classification_report"],
        "confusion_matrix": window_metrics["confusion_matrix"],
        "recording_level": {
            **recording_metrics,
        },
        "recording_aggregation_comparison": aggregation_metrics,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_cm = np.asarray(window_metrics["confusion_matrix"])
    recording_cm = np.asarray(recording_metrics["confusion_matrix"])
    save_confusion_matrix(window_cm, output_dir / "confusion_matrix.png")
    save_confusion_matrix(
        recording_cm, output_dir / "confusion_matrix_recording_level.png"
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    with (output_dir / "recording_aggregation_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregation_metrics, handle, indent=2)
    comparison_rows = [
        "| Aggregation method | Recording accuracy | Macro F1 | Weighted F1 |",
        "|---|---:|---:|---:|",
    ]
    for method, values in aggregation_metrics.items():
        display_name = values["aggregation"]
        if method == "top_confidence_mean":
            display_name = f"Top-{args.top_confidence_fraction:.0%} confidence average"
        comparison_rows.append(
            f"| {display_name} | {values['accuracy']:.2%} | "
            f"{values['macro_f1']:.2%} | {values['weighted_f1']:.2%} |"
        )
    (output_dir / "recording_aggregation_comparison.md").write_text(
        "\n".join(comparison_rows) + "\n", encoding="utf-8"
    )
    print("Window-level metrics on held-out recording rows")
    print(window_metrics["classification_report"])
    print(f"Accuracy: {window_metrics['accuracy']:.4%}")
    print(f"Macro F1: {window_metrics['macro_f1']:.4%}")
    print(f"Weighted F1: {window_metrics['weighted_f1']:.4%}")
    print("True distribution:", window_metrics["true_distribution"])
    print("Predicted distribution:", window_metrics["predicted_distribution"])
    print("Per-class precision / recall / F1:")
    for class_name in CLASS_NAMES:
        values = window_metrics["per_class"][class_name]
        print(
            f"  {class_name}: {values['precision']:.4%} / "
            f"{values['recall']:.4%} / {values['f1']:.4%}"
        )
    print("Confusion matrix:")
    print(window_cm)
    print("Recording-level metrics (mean window logits per source row)")
    print(recording_metrics["classification_report"])
    print(f"Accuracy: {recording_metrics['accuracy']:.4%}")
    print(f"Macro F1: {recording_metrics['macro_f1']:.4%}")
    print(f"Weighted F1: {recording_metrics['weighted_f1']:.4%}")
    print("True distribution:", recording_metrics["true_distribution"])
    print("Predicted distribution:", recording_metrics["predicted_distribution"])
    print("Recording-level confusion matrix:")
    print(recording_cm)
    print("Recording aggregation comparison:")
    print("\n".join(comparison_rows))
    print("Recording-row overlap: 0")


if __name__ == "__main__":
    main()
