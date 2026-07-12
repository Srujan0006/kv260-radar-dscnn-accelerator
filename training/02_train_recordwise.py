#!/usr/bin/env python3
"""Train a DPU-friendly radar CNN using pre-separated recording rows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import (
    CachedSpectrogramDataset,
    RadarSpecAugment,
    load_spectrogram_cache,
)
from radar2dcnn.labels import CLASS_NAMES, class_distribution
from radar2dcnn.imbalance import (
    IMBALANCE_STRATEGIES,
    compute_class_weights,
    make_weighted_sampler,
)
from radar2dcnn.metrics import compute_metrics
from radar2dcnn.model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Make Python/NumPy augmentation deterministic in DataLoader workers."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)
    )
    return float(np.mean(losses)), metrics


def cache_row_indices(path: Path) -> set[int]:
    with np.load(path, allow_pickle=False) as data:
        if "row_indices" not in data.files:
            raise KeyError(f"{path} has no row_indices; rebuild it with record-wise preprocessing")
        return set(data["row_indices"].astype(int).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default="outputs/recordwise/train/spectrogram_cache.npz")
    parser.add_argument("--val-data", default="outputs/recordwise/val/spectrogram_cache.npz")
    parser.add_argument("--test-data", default="outputs/recordwise/test/spectrogram_cache.npz")
    parser.add_argument("--output-dir", default="outputs/recordwise")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=["radar2dcnn", "radar_dscnn2d"],
        default="radar_dscnn2d",
        help="Model used for this controlled experiment",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable RadarSpecAugment for training samples only",
    )
    parser.add_argument(
        "--imbalance-strategy",
        choices=IMBALANCE_STRATEGIES,
        default="inverse",
        help=(
            "inverse reproduces the strong baseline; sampler_only uses balanced "
            "sampling with normal cross entropy; capped limits inverse weights"
        ),
    )
    parser.add_argument(
        "--max-class-weight",
        type=float,
        default=8.0,
        help="Maximum loss weight used by --imbalance-strategy capped",
    )
    parser.add_argument(
        "--effective-beta",
        type=float,
        default=0.999,
        help="Beta used by effective-number class weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path, test_path = map(
        Path, (args.train_data, args.val_data, args.test_data)
    )
    train_rows = cache_row_indices(train_path)
    val_rows = cache_row_indices(val_path)
    test_rows = cache_row_indices(test_path)
    if not (
        train_rows.isdisjoint(val_rows)
        and train_rows.isdisjoint(test_rows)
        and val_rows.isdisjoint(test_rows)
    ):
        raise RuntimeError("Recording-row leakage detected between caches")

    train_arrays = load_spectrogram_cache(train_path)
    val_arrays = load_spectrogram_cache(val_path)
    train_transform = RadarSpecAugment() if args.augmentation else None
    train_ds = CachedSpectrogramDataset(train_arrays, transform=train_transform)
    val_ds = CachedSpectrogramDataset(val_arrays)

    counts = np.bincount(train_arrays.y, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise RuntimeError(f"Training split is missing a class: counts={counts.tolist()}")
    class_weights = compute_class_weights(
        counts,
        strategy=args.imbalance_strategy,
        max_weight=args.max_class_weight,
        effective_beta=args.effective_beta,
    )
    sampler = (
        make_weighted_sampler(train_arrays.y, counts, args.seed)
        if args.imbalance_strategy == "sampler_only"
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        num_classes=len(CLASS_NAMES),
        base_channels=args.base_channels,
    ).to(device)
    loss_weights = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    train_criterion = nn.CrossEntropyLoss(weight=loss_weights)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.scheduler_patience, factor=0.5
    )
    print(
        f"Device: {device}; model: {args.model}; "
        f"trainable parameters: {count_parameters(model):,}"
    )
    print(f"Train-only augmentation: {args.augmentation}")
    print("Train distribution:", class_distribution(train_arrays.y))
    print("Validation distribution:", class_distribution(val_arrays.y))
    print(f"Imbalance strategy: {args.imbalance_strategy}")
    print(
        "Cross-entropy class weights:",
        None if class_weights is None else class_weights.tolist(),
    )

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = out_dir / "best_model.pth"
    manifest_path = out_dir / "split_manifest.json"
    split_strategy = "recording_rows"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            split_strategy = json.load(handle).get("strategy", split_strategy)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, train_criterion, optimizer, device, train=True
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, val_criterion, optimizer, device, train=False
        )
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "architecture": args.model,
                        "num_classes": len(CLASS_NAMES),
                        "base_channels": args.base_channels,
                    },
                    "class_names": CLASS_NAMES,
                    "best_val_macro_f1": best_macro_f1,
                    "best_val_accuracy": val_metrics["accuracy"],
                    "epoch": epoch,
                    "split_strategy": split_strategy,
                    "imbalance_strategy": args.imbalance_strategy,
                    "class_weights": (
                        None if class_weights is None else class_weights.tolist()
                    ),
                    "max_class_weight": args.max_class_weight,
                    "effective_beta": args.effective_beta,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        results = {
            "status": "training",
            "split_strategy": split_strategy,
            "recording_row_overlap": 0,
            "imbalance_strategy": args.imbalance_strategy,
            "model": args.model,
            "train_only_augmentation": args.augmentation,
            "class_weights": (
                None
                if class_weights is None
                else {
                    CLASS_NAMES[i]: float(class_weights[i])
                    for i in range(len(CLASS_NAMES))
                }
            ),
            "max_class_weight": args.max_class_weight,
            "effective_beta": args.effective_beta,
            "selection_metric": "validation_macro_f1",
            "device": str(device),
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "train_distribution": class_distribution(train_arrays.y),
            "val_distribution": class_distribution(val_arrays.y),
            "history": history,
        }
        with (out_dir / "train_results.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    results["status"] = "complete"
    results["epochs_completed"] = len(history)
    results["checkpoint"] = str(checkpoint_path)
    with (out_dir / "train_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Best validation macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
