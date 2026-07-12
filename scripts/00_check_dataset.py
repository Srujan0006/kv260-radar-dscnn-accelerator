#!/usr/bin/env python3
"""Check radar dataset shape, labels, and a few spectrogram conversions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import load_radar_iq_npz
from radar2dcnn.labels import CLASS_NAMES, class_distribution
from radar2dcnn.spectrogram import iq_to_spectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .npz/.npy radar IQ dataset")
    parser.add_argument("--num-check", type=int, default=5)
    parser.add_argument("--range-mode", choices=["center", "mean", "max"], default="center")
    args = parser.parse_args()

    arrays = load_radar_iq_npz(args.input)
    print(f"Loaded: {args.input}")
    print(f"X shape: {arrays.X.shape}, dtype={arrays.X.dtype}")
    print(f"y shape: {arrays.y.shape}, dtype={arrays.y.dtype}")
    print("Class distribution:")
    for name, count in class_distribution(arrays.y).items():
        print(f"  {name:16s}: {count}")

    n = min(args.num_check, len(arrays.y))
    print(f"\nChecking first {n} spectrogram conversions...")
    for i in range(n):
        img = iq_to_spectrogram(arrays.X[i], image_size=100, range_mode=args.range_mode)
        print(
            f"  sample {i}: label={CLASS_NAMES[int(arrays.y[i])]}, "
            f"image={img.shape}, min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}"
        )

    print("\nDataset check complete.")


if __name__ == "__main__":
    main()
