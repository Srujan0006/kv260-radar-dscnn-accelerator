#!/usr/bin/env python3
"""CPU/GPU demo fallback for the trained 2D CNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.functional import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import CachedSpectrogramDataset, load_spectrogram_cache
from radar2dcnn.labels import CLASS_NAMES
from radar2dcnn.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    arrays = load_spectrogram_cache(args.data)
    ds = CachedSpectrogramDataset(arrays)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint).to(device)
    model.eval()

    correct = 0
    total = min(args.num_samples, len(ds))
    with torch.no_grad():
        for i in range(total):
            x, y = ds[i]
            logits = model(x.unsqueeze(0).to(device))
            probs = softmax(logits, dim=1)[0].cpu()
            pred = int(torch.argmax(probs).item())
            correct += int(pred == y)
            print(
                f"sample={i:04d} true={CLASS_NAMES[y]:16s} pred={CLASS_NAMES[pred]:16s} "
                f"conf={float(probs[pred]):.3f}"
            )
    print(f"Demo accuracy on {total} samples: {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
