#!/usr/bin/env python3
"""Generate a tiny synthetic radar-like IQ dataset for pipeline testing only.

This data is NOT scientifically valid. It is only for checking that scripts run end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_signal(cls: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(256, dtype=np.float32)
    cells = []
    for r in range(5):
        noise = 0.15 * (rng.standard_normal(256) + 1j * rng.standard_normal(256))
        amp = 1.0 + 0.1 * rng.standard_normal()
        if cls == 0:  # drone: multiple rotor-like tones
            sig = amp * (
                np.exp(1j * 2 * np.pi * 0.08 * t)
                + 0.6 * np.exp(1j * 2 * np.pi * 0.17 * t)
                + 0.4 * np.exp(1j * 2 * np.pi * 0.27 * t)
            )
        elif cls == 1:  # bird: one wobbling tone
            sig = amp * np.exp(1j * (2 * np.pi * 0.05 * t + 0.8 * np.sin(2 * np.pi * 0.015 * t)))
        elif cls == 2:  # human: strong near-zero plus sideband
            sig = amp * (1.2 + 0.3 * np.exp(1j * 2 * np.pi * 0.025 * t))
        else:  # corner reflector: mostly static
            sig = amp * np.ones_like(t, dtype=np.complex64)
        range_scale = 1.0 / (1.0 + abs(r - 2) * 0.35)
        cells.append((range_scale * sig + noise).astype(np.complex64))
    return np.stack(cells, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/toy_radar_iq.npz")
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = [], []
    for cls in range(4):
        for _ in range(args.samples_per_class):
            X.append(make_signal(cls, rng))
            y.append(cls)
    X = np.asarray(X, dtype=np.complex64)
    y = np.asarray(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y)
    print(f"Saved toy dataset: {out}")
    print(f"X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
