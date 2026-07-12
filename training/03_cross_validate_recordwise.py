#!/usr/bin/env python3
"""Run nested grouped-stratified cross-validation on record-wise caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.cross_validation import make_nested_grouped_folds
from radar2dcnn.imbalance import IMBALANCE_STRATEGIES
from radar2dcnn.labels import CLASS_NAMES

CACHE_FIELDS = (
    "images",
    "y",
    "sample_ids",
    "row_indices",
    "record_ids",
    "window_starts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root", default="outputs/recordwise_topk3_stride32"
    )
    parser.add_argument(
        "--output-dir", default="outputs/recordwise_topk3_stride32_cv5_capped"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--imbalance-strategy",
        choices=IMBALANCE_STRATEGIES,
        default="capped",
    )
    parser.add_argument("--max-class-weight", type=float, default=8.0)
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and audit fold caches without training",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse folds that already contain test_metrics.json",
    )
    return parser.parse_args()


def load_combined_cache(cache_root: Path) -> tuple[dict[str, np.ndarray], dict]:
    parts: list[dict[str, np.ndarray]] = []
    static: dict | None = None
    for split in ("train", "val", "test"):
        path = cache_root / split / "spectrogram_cache.npz"
        with np.load(path, allow_pickle=False) as cache:
            missing = set(CACHE_FIELDS) - set(cache.files)
            if missing:
                raise KeyError(f"{path} is missing cache fields: {sorted(missing)}")
            parts.append({field: cache[field] for field in CACHE_FIELDS})
            if static is None:
                static = {
                    "class_names": cache["class_names"],
                    "image_size": cache["image_size"],
                    "range_mode": cache["range_mode"],
                    "top_k": cache["top_k"],
                }
    if static is None:
        raise ValueError(f"No caches found under {cache_root}")
    combined = {
        field: np.concatenate([part[field] for part in parts], axis=0)
        for field in CACHE_FIELDS
    }
    if len(np.unique(combined["sample_ids"])) != len(combined["sample_ids"]):
        raise ValueError("Combined caches contain duplicate sample IDs")
    return combined, static


def save_fold_cache(
    path: Path,
    combined: dict[str, np.ndarray],
    static: dict,
    indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{field: combined[field][indices] for field in CACHE_FIELDS},
        **static,
    )


def split_summary(labels: np.ndarray, rows: np.ndarray) -> dict:
    window_counts = Counter(labels.tolist())
    row_labels: dict[int, int] = {}
    for row, label in zip(rows.tolist(), labels.tolist()):
        if row in row_labels and row_labels[row] != label:
            raise ValueError(f"Source row {row} contains multiple labels")
        row_labels[row] = label
    row_counts = Counter(row_labels.values())
    return {
        "windows": int(len(labels)),
        "source_rows": int(len(row_labels)),
        "window_class_distribution": {
            name: int(window_counts.get(index, 0))
            for index, name in enumerate(CLASS_NAMES)
        },
        "recording_class_distribution": {
            name: int(row_counts.get(index, 0))
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def aggregate_results(output_dir: Path, fold_count: int) -> dict:
    fold_results = []
    for fold_number in range(1, fold_count + 1):
        metrics_path = output_dir / f"fold_{fold_number}" / "test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold_results.append(
            {
                "fold": fold_number,
                "window_accuracy": metrics["test_accuracy"],
                "window_macro_f1": metrics["macro_f1"],
                "window_weighted_f1": metrics["weighted_f1"],
                **{
                    f"recording_{method}_{metric_name}": values[metric_name]
                    for method, values in metrics[
                        "recording_aggregation_comparison"
                    ].items()
                    for metric_name in ("accuracy", "macro_f1", "weighted_f1")
                },
            }
        )

    metric_names = [key for key in fold_results[0] if key != "fold"]
    summary = {}
    for name in metric_names:
        values = np.asarray([fold[name] for fold in fold_results], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return {"folds": fold_results, "summary": summary}


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, static = load_combined_cache(cache_root)
    folds = make_nested_grouped_folds(
        combined["y"],
        combined["row_indices"],
        n_splits=args.folds,
        inner_splits=args.inner_folds,
        seed=args.seed,
    )

    fold_manifests = []
    for fold_number, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
        fold_dir = output_dir / f"fold_{fold_number}"
        index_by_split = {"train": train_idx, "val": val_idx, "test": test_idx}
        summaries = {}
        for split, indices in index_by_split.items():
            cache_path = fold_dir / split / "spectrogram_cache.npz"
            save_fold_cache(cache_path, combined, static, indices)
            summaries[split] = split_summary(
                combined["y"][indices], combined["row_indices"][indices]
            )
        manifest = {
            "strategy": "nested_stratified_group_kfold",
            "outer_fold": fold_number,
            "outer_folds": args.folds,
            "inner_folds": args.inner_folds,
            "seed": args.seed,
            "group_key": "source_row_index",
            "recording_row_overlap": 0,
            "splits": summaries,
        }
        (fold_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        fold_manifests.append(manifest)

        metrics_path = fold_dir / "test_metrics.json"
        if args.prepare_only or (args.skip_existing and metrics_path.exists()):
            continue
        train_cache = str(fold_dir / "train" / "spectrogram_cache.npz")
        val_cache = str(fold_dir / "val" / "spectrogram_cache.npz")
        test_cache = str(fold_dir / "test" / "spectrogram_cache.npz")
        common = [
            "--train-data", train_cache,
            "--val-data", val_cache,
            "--test-data", test_cache,
        ]
        run_command(
            [
                sys.executable,
                "training/02_train_recordwise.py",
                *common,
                "--output-dir", str(fold_dir),
                "--model", "radar_dscnn2d",
                "--augmentation",
                "--imbalance-strategy", args.imbalance_strategy,
                "--max-class-weight", str(args.max_class_weight),
                "--effective-beta", str(args.effective_beta),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--patience", str(args.patience),
                "--scheduler-patience", str(args.scheduler_patience),
                "--num-workers", str(args.num_workers),
                "--seed", str(args.seed + fold_number - 1),
            ]
        )
        run_command(
            [
                sys.executable,
                "evaluation/03_eval_recordwise.py",
                "--data", test_cache,
                "--train-data", train_cache,
                "--val-data", val_cache,
                "--checkpoint", str(fold_dir / "best_model.pth"),
                "--output-dir", str(fold_dir),
                "--batch-size", str(args.batch_size),
            ]
        )

    (output_dir / "fold_manifests.json").write_text(
        json.dumps(fold_manifests, indent=2), encoding="utf-8"
    )
    if args.prepare_only:
        print(f"Prepared and audited {args.folds} folds in {output_dir}")
        return

    results = aggregate_results(output_dir, args.folds)
    results["configuration"] = {
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "imbalance_strategy": args.imbalance_strategy,
        "max_class_weight": args.max_class_weight,
    }
    (output_dir / "cross_validation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    for name, values in results["summary"].items():
        print(f"{name}: {values['mean']:.4%} ± {values['std']:.4%}")


if __name__ == "__main__":
    main()
