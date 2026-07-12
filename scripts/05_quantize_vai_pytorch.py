#!/usr/bin/env python3
"""Quantize Radar2DCNN with Vitis AI PyTorch quantizer.

Run inside the Vitis AI PyTorch Docker/container:

    conda activate vitis-ai-pytorch
    python scripts/05_quantize_vai_pytorch.py --quant-mode calib ...
    python scripts/05_quantize_vai_pytorch.py --quant-mode test ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import CachedSpectrogramDataset, load_spectrogram_cache
from radar2dcnn.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Cached spectrogram .npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/vai_quant")
    parser.add_argument("--quant-mode", choices=["calib", "test"], default="calib")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calib-samples", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    try:
        from pytorch_nndct.apis import torch_quantizer
    except Exception as exc:
        raise RuntimeError(
            "pytorch_nndct is not available. Run this inside Vitis AI PyTorch Docker "
            "after `conda activate vitis-ai-pytorch`."
        ) from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_spectrogram_cache(args.data)
    ds = CachedSpectrogramDataset(arrays)
    limit = min(args.calib_samples, len(ds))
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(len(ds), size=limit, replace=False)
    loader = DataLoader(Subset(ds, subset_idx), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint).to(device)
    model.eval()

    input_args = torch.randn(1, 1, 100, 100).to(device)
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=input_args,
        output_dir=str(out_dir),
    )
    qmodel = quantizer.quant_model

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"VAI {args.quant_mode}"):
            x = x.to(device)
            y = y.to(device)
            logits = qmodel(x)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

    print(f"Quantized model run accuracy on subset: {correct / max(total, 1):.4f}")

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print(f"Exported quant config to: {out_dir}")
    else:
        quantizer.export_xmodel(output_dir=str(out_dir), deploy_check=False)
        print(f"Exported xmodel to: {out_dir}")
        print("Expected file name is usually Radar2DCNN_int.xmodel or similar.")


if __name__ == "__main__":
    main()
