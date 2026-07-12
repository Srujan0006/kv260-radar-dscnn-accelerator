#!/usr/bin/env python3
"""Evaluate a trained Radar2DCNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import CachedSpectrogramDataset, load_spectrogram_cache, make_splits
from radar2dcnn.labels import CLASS_NAMES
from radar2dcnn.metrics import compute_metrics, save_confusion_matrix
from radar2dcnn.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/eval_run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=["test", "val", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_spectrogram_cache(args.data)
    if args.split == "all":
        idx = None
    else:
        train_idx, val_idx, test_idx = make_splits(arrays.y, seed=args.seed)
        idx = val_idx if args.split == "val" else test_idx

    ds = CachedSpectrogramDataset(arrays, idx)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint).to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluate"):
            logits = model(x.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    print(metrics["report"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    save_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "classification_report": metrics["report"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
