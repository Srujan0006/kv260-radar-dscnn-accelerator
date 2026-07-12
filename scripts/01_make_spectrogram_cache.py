#!/usr/bin/env python3
"""Convert raw IQ dataset to cached 100x100 spectrogram dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import load_radar_iq_npz
from radar2dcnn.spectrogram import iq_to_spectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw radar IQ .npz/.npy")
    parser.add_argument("--output", required=True, help="Output cached spectrogram .npz")
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--range-mode", choices=["center", "mean", "max"], default="center")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for debugging")
    args = parser.parse_args()

    arrays = load_radar_iq_npz(args.input)
    X = arrays.X
    y = arrays.y
    if args.limit is not None:
        X = X[: args.limit]
        y = y[: args.limit]

    images = np.zeros((len(y), 1, args.image_size, args.image_size), dtype=np.float32)
    for i in tqdm(range(len(y)), desc="IQ -> spectrogram"):
        images[i, 0] = iq_to_spectrogram(
            X[i], image_size=args.image_size, range_mode=args.range_mode
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        y=y.astype(np.int64),
        image_size=np.array(args.image_size),
        range_mode=np.array(args.range_mode),
    )
    print(f"Saved cache: {out_path}")
    print(f"images shape: {images.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    main()
