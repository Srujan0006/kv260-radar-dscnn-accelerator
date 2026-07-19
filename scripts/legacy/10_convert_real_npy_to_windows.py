#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np


def get_record_id(value):
    """Extract ID string from object-array field."""
    if isinstance(value, np.ndarray):
        return str(value.flat[0])
    return str(value)


def choose_top_range_bins(iq_window, num_bins=5):
    """
    iq_window shape: [range_bins, time]
    Select strongest range bins based on average energy.
    """
    energy = np.mean(np.abs(iq_window) ** 2, axis=1)
    top = np.argsort(energy)[-num_bins:]
    top = np.sort(top)
    return iq_window[top, :]


def majority_label(values):
    values = np.asarray(values).reshape(-1).astype(int)
    if len(values) == 0:
        return 0
    counts = Counter(values.tolist())
    return counts.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../radar_data/radar_iq.npy")
    parser.add_argument("--output", default="data/real/radar_iq_windows.npz")
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--range-bins", type=int, default=5)
    parser.add_argument(
        "--label-source",
        choices=["prefix", "col4_majority", "col5_majority"],
        default="prefix",
        help="Use ID prefix or majority label from column 4/5.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(input_path, allow_pickle=True)

    print("Loaded:", input_path)
    print("Original shape:", data.shape)
    print("Original dtype:", data.dtype)

    # Build prefix label map if needed.
    prefixes = []
    for row in data:
        rec_id = get_record_id(row[0])
        prefixes.append(rec_id[0])

    unique_prefixes = sorted(set(prefixes))
    prefix_to_label = {p: i for i, p in enumerate(unique_prefixes)}

    print("Detected prefixes:", unique_prefixes)
    print("Prefix label map:", prefix_to_label)

    X_samples = []
    y_samples = []
    sample_ids = []

    for row_index, row in enumerate(data):
        rec_id = get_record_id(row[0])
        iq = np.asarray(row[1])

        if iq.ndim != 2:
            print(f"Skipping row {row_index}: IQ ndim is {iq.ndim}, expected 2")
            continue

        # Expected: [range_bins, time_samples]
        range_count, time_count = iq.shape

        if time_count < args.window:
            print(f"Skipping row {row_index}: only {time_count} time samples")
            continue

        for start in range(0, time_count - args.window + 1, args.stride):
            end = start + args.window

            iq_window = iq[:, start:end]  # [1280, 256]
            selected = choose_top_range_bins(iq_window, num_bins=args.range_bins)

            # selected shape: [5, 256]
            if selected.shape != (args.range_bins, args.window):
                continue

            if args.label_source == "prefix":
                label = prefix_to_label[rec_id[0]]
            elif args.label_source == "col4_majority":
                label = majority_label(row[4][start:end])
            elif args.label_source == "col5_majority":
                label = majority_label(row[5][start:end])
            else:
                raise ValueError(args.label_source)

            X_samples.append(selected.astype(np.complex64))
            y_samples.append(int(label))
            sample_ids.append(f"{rec_id}_{start}_{end}")

    X = np.stack(X_samples, axis=0)
    y = np.asarray(y_samples, dtype=np.int64)
    sample_ids = np.asarray(sample_ids)

    np.savez_compressed(output_path, X=X, y=y, sample_ids=sample_ids)

    label_map_path = output_path.with_suffix(".label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_source": args.label_source,
                "prefix_to_label": prefix_to_label,
                "note": "If label_source is prefix, labels are assigned from first character of record ID.",
            },
            f,
            indent=2,
        )

    print("\nSaved:", output_path)
    print("Saved label map:", label_map_path)
    print("X shape:", X.shape, X.dtype)
    print("y shape:", y.shape, y.dtype)
    print("Label distribution:", Counter(y.tolist()))


if __name__ == "__main__":
    main()
