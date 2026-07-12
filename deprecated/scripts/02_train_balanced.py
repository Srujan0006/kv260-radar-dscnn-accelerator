#!/usr/bin/env python3
"""Train Radar2DCNN on cached spectrograms."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import CachedSpectrogramDataset, load_spectrogram_cache, make_splits
from radar2dcnn.labels import CLASS_NAMES, class_distribution
from radar2dcnn.metrics import compute_metrics, save_confusion_matrix
from radar2dcnn.model import Radar2DCNN, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses = []
    y_true, y_pred = [], []
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred)
    return float(np.mean(losses)), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Cached spectrogram .npz")
    parser.add_argument("--output-dir", default="outputs/train_run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_spectrogram_cache(args.data)
    train_idx, val_idx, test_idx = make_splits(arrays.y, args.val_ratio, args.test_ratio, args.seed)

    np.savez(out_dir / "splits.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print("Class distribution:", class_distribution(arrays.y))
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = CachedSpectrogramDataset(arrays, train_idx)
    val_ds = CachedSpectrogramDataset(arrays, val_idx)
    test_ds = CachedSpectrogramDataset(arrays, test_idx)

    
    # Balanced sampler: makes minority classes appear more often during training
    train_labels = arrays.y[train_idx]
    class_counts = np.bincount(train_labels)
    class_counts = np.maximum(class_counts, 1)

    sample_weights = 1.0 / class_counts[train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Radar2DCNN(num_classes=len(CLASS_NAMES), base_channels=args.base_channels).to(device)
    print(f"Using device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=arrays.y[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    history = []
    best_val_acc = -1.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_metrics["accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {"num_classes": len(CLASS_NAMES), "base_channels": args.base_channels},
                    "class_names": CLASS_NAMES,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # Final test with best checkpoint.
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print("\nFinal test report:")
    print(test_metrics["report"])
    save_confusion_matrix(test_metrics["confusion_matrix"], out_dir / "confusion_matrix.png")
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "test_accuracy": test_metrics["accuracy"],
                "classification_report": test_metrics["report"],
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
