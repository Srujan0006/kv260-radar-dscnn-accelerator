#!/usr/bin/env python3
"""Build leakage-free caches by assigning complete source recording rows.

Source row identity is the grouping key because display IDs such as ``CR`` and
``D1`` are repeated in the source table.  Rows are split before windowing, so
every overlapping window from one source recording remains in exactly one set.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.labels import CLASS_NAMES, label_to_superclass
from radar2dcnn.spectrogram import iq_to_spectrogram, topk_energy_mean_spectrogram

def get_record_id(value: object) -> str:
    """Return the scalar recording ID stored in the object-array field."""
    if isinstance(value, np.ndarray):
        return str(value.flat[0])
    return str(value)


def record_label(record_id: str) -> int:
    """Map source recording names, including compact D1..D6 IDs, to 0..3."""
    if record_id.strip().lower().startswith("d"):
        return 0  # Drone
    return label_to_superclass(record_id)


def choose_top_range_bins(iq_window: np.ndarray, num_bins: int) -> np.ndarray:
    """Select the strongest range bins in stable range-index order."""
    energy = np.mean(np.abs(iq_window) ** 2, axis=1)
    indices = np.sort(np.argsort(energy)[-num_bins:])
    return iq_window[indices, :].astype(np.complex64, copy=False)


def make_stratified_recording_split(
    row_indices: np.ndarray,
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[int, str]:
    """Assign entire recording rows with class stratification at both stages."""
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Train/val/test ratios must be positive and sum to 1")

    train_rows, heldout_rows, _, heldout_labels = train_test_split(
        row_indices,
        labels,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=labels,
    )
    test_fraction = test_ratio / (val_ratio + test_ratio)
    val_rows, test_rows = train_test_split(
        heldout_rows,
        test_size=test_fraction,
        random_state=seed,
        stratify=heldout_labels,
    )

    split_by_row = {int(row): "train" for row in train_rows}
    split_by_row.update({int(row): "val" for row in val_rows})
    split_by_row.update({int(row): "test" for row in test_rows})
    if len(split_by_row) != len(row_indices):
        raise AssertionError("A recording row was lost or duplicated during splitting")
    return split_by_row


def make_modulo_recording_split(row_indices: list[int]) -> dict[int, str]:
    """Apply the fixed scientific split: row 0 train, row 1 val, row 2 test."""
    split_by_remainder = {0: "train", 1: "val", 2: "test"}
    return {
        int(row_index): split_by_remainder[int(row_index) % 3]
        for row_index in row_indices
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="../radar_data/radar_iq.npy")
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", default="outputs/recordwise"
    )
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--range-bins", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument(
        "--split-strategy",
        "--split",
        dest="split_strategy",
        choices=["row_modulo", "class_stratified", "grouped_stratified"],
        default="grouped_stratified",
        help=(
            "row_modulo applies source row 0->train, 1->val, 2->test cyclically; "
            "grouped_stratified/class_stratified use a stratified 70/15/15 split "
            "while keeping every source row in one set"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--range-mode",
        choices=["center", "mean", "max", "topk_energy_mean"],
        default="max",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of energy-ranked range-bin spectrograms to average",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window != 256:
        raise ValueError(
            "The current spectrogram model expects --window 256"
        )
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.range_bins <= 0:
        raise ValueError("--range-bins must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    source = np.load(args.input, allow_pickle=True)
    if source.ndim != 2 or source.shape[1] < 2:
        raise ValueError(f"Expected a recording table [N, >=2], got {source.shape}")

    # Rows too short to produce even one window are not model examples and are
    # excluded before stratification.  This keeps the requested ratios tied to
    # the recordings that actually enter training or evaluation.
    eligible_rows: list[int] = []
    eligible_labels: list[int] = []
    for row_index, row in enumerate(source):
        iq = np.asarray(row[1])
        if iq.ndim != 2:
            raise ValueError(f"Row {row_index} IQ must be 2-D, got {iq.shape}")
        if iq.shape[1] >= args.window:
            eligible_rows.append(row_index)
            eligible_labels.append(record_label(get_record_id(row[0])))

    if args.split_strategy == "row_modulo":
        split_by_row = make_modulo_recording_split(eligible_rows)
        strategy_metadata = {
            "strategy": "row_index_modulo_3",
            "assignment": {"0": "train", "1": "val", "2": "test"},
        }
    else:
        split_by_row = make_stratified_recording_split(
            np.asarray(eligible_rows, dtype=np.int64),
            np.asarray(eligible_labels, dtype=np.int64),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
        strategy_metadata = {
            "strategy": "grouped_stratified_recording_rows",
            "ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "seed": args.seed,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_data = {
        name: {
            "images": [],
            "y": [],
            "sample_ids": [],
            "row_indices": [],
            "record_ids": [],
            "window_starts": [],
        }
        for name in ("train", "val", "test")
    }

    for row_index in tqdm(eligible_rows, desc="Recording rows"):
        row = source[row_index]
        split = split_by_row[row_index]
        rec_id = get_record_id(row[0])
        label = record_label(rec_id)
        iq = np.asarray(row[1])
        time_count = iq.shape[1]
        for start in range(0, time_count - args.window + 1, args.stride):
            iq_window = iq[:, start : start + args.window]
            if args.range_mode == "topk_energy_mean":
                if args.top_k > iq_window.shape[0]:
                    raise ValueError(
                        f"--top-k {args.top_k} exceeds row {row_index}'s "
                        f"{iq_window.shape[0]} range bins"
                    )
                image = topk_energy_mean_spectrogram(
                    iq_window, top_k=args.top_k, image_size=args.image_size
                )
            else:
                if args.range_bins > iq_window.shape[0]:
                    raise ValueError(
                        f"--range-bins {args.range_bins} exceeds row {row_index}'s "
                        f"{iq_window.shape[0]} range bins"
                    )
                selected = choose_top_range_bins(iq_window, args.range_bins)
                if selected.shape[0] != 5:
                    raise ValueError(
                        "center/mean/max modes require --range-bins 5; use "
                        "topk_energy_mean for an arbitrary top-k selection"
                    )
                image = iq_to_spectrogram(
                    selected, image_size=args.image_size, range_mode=args.range_mode
                )
            sample_id = f"row{row_index:03d}:{rec_id}:{start}:{start + args.window}"
            target = split_data[split]
            target["images"].append(image[None, :, :])
            target["y"].append(label)
            target["sample_ids"].append(sample_id)
            target["row_indices"].append(row_index)
            target["record_ids"].append(rec_id)
            target["window_starts"].append(start)

    summary = {
        "source": str(Path(args.input)),
        "group_key": "source_row_index",
        "source_rows_total": int(len(source)),
        "source_rows_eligible": int(len(eligible_rows)),
        "source_rows_too_short": int(len(source) - len(eligible_rows)),
        "window": args.window,
        "stride": args.stride,
        "range_bins": args.range_bins,
        "image_size": args.image_size,
        "range_mode": args.range_mode,
        "top_k": args.top_k if args.range_mode == "topk_energy_mean" else None,
        "splits": {},
    }
    summary.update(strategy_metadata)
    row_sets: dict[str, set[int]] = {}

    for split in ("train", "val", "test"):
        values = split_data[split]
        if not values["images"]:
            raise RuntimeError(f"Split {split!r} contains no windows")
        images = np.stack(values["images"]).astype(np.float32, copy=False)
        labels = np.asarray(values["y"], dtype=np.int64)
        row_indices = np.asarray(values["row_indices"], dtype=np.int64)
        row_sets[split] = set(row_indices.tolist())
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_dir / "spectrogram_cache.npz"
        np.savez_compressed(
            output_path,
            images=images,
            y=labels,
            sample_ids=np.asarray(values["sample_ids"]),
            row_indices=row_indices,
            record_ids=np.asarray(values["record_ids"]),
            window_starts=np.asarray(values["window_starts"], dtype=np.int64),
            class_names=np.asarray(CLASS_NAMES),
            image_size=np.asarray(args.image_size),
            range_mode=np.asarray(args.range_mode),
            top_k=np.asarray(
                args.top_k if args.range_mode == "topk_energy_mean" else 0
            ),
        )
        counts = Counter(labels.tolist())
        row_label_by_index = dict(zip(eligible_rows, eligible_labels))
        row_class_counts = Counter(row_label_by_index[index] for index in row_sets[split])
        summary["splits"][split] = {
            "windows": int(len(labels)),
            "source_rows_with_windows": int(len(row_sets[split])),
            "class_distribution": {
                CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))
            },
            "recording_class_distribution": {
                CLASS_NAMES[i]: int(row_class_counts.get(i, 0))
                for i in range(len(CLASS_NAMES))
            },
        }
        print(f"Saved {output_path}: {images.shape}")

    if not (
        row_sets["train"].isdisjoint(row_sets["val"])
        and row_sets["train"].isdisjoint(row_sets["test"])
        and row_sets["val"].isdisjoint(row_sets["test"])
    ):
        raise AssertionError("A source recording row appears in more than one split")
    summary["recording_row_overlap"] = 0

    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
