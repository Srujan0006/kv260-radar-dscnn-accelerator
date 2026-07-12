#!/usr/bin/env python3
"""Compare the host-float HLS kernel with the folded PyTorch network.

This verifies layer ordering, padding, pooling, weight export, and BN folding.
It is not a bit-accurate fixed-point test; use Vitis HLS for that final check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ACCEL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radar2dcnn.model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "outputs/final_model/radar_dscnn2d_topk3_capped8_best.pth"
        ),
    )
    parser.add_argument(
        "--cache",
        default=str(
            PROJECT_ROOT
            / "outputs/recordwise_topk3_stride32/test/spectrogram_cache.npz"
        ),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as data:
        images = data["images"]
        labels = data["y"]
        sample_ids = data["sample_ids"] if "sample_ids" in data.files else None
        if not 0 <= args.sample_index < len(images):
            raise IndexError(f"sample-index must be in [0, {len(images) - 1}]")
        image = np.asarray(images[args.sample_index], dtype=np.float32).reshape(100, 100)
        label = int(labels[args.sample_index])
        sample_id = (
            str(sample_ids[args.sample_index])
            if sample_ids is not None
            else str(args.sample_index)
        )

    model = load_checkpoint(args.checkpoint, map_location="cpu")
    with torch.no_grad():
        pytorch_logits = model(torch.from_numpy(image[None, None])).numpy()[0]

    hls_dir = ACCEL_ROOT / "hls"
    with tempfile.TemporaryDirectory(prefix="radar_hls_compare_") as temp:
        temp_dir = Path(temp)
        binary = temp_dir / "radar_cnn_host"
        input_path = temp_dir / "input.txt"
        np.savetxt(input_path, image.reshape(-1), fmt="%.9g")
        subprocess.run(
            [
                args.compiler,
                "-std=c++14",
                "-O2",
                "-DRADAR_CNN_SOFTWARE_FLOAT",
                str(hls_dir / "radar_cnn_hls.cpp"),
                str(hls_dir / "tb_radar_cnn_hls.cpp"),
                "-I",
                str(hls_dir),
                "-o",
                str(binary),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(binary), str(input_path)],
            check=True,
            text=True,
            capture_output=True,
        )

    hls_logits = np.asarray(
        [
            float(line.split()[2])
            for line in completed.stdout.splitlines()
            if line.startswith("LOGIT ")
        ],
        dtype=np.float32,
    )
    if hls_logits.shape != (4,):
        raise RuntimeError(f"Could not parse four HLS logits:\n{completed.stdout}")
    abs_error = np.abs(pytorch_logits - hls_logits)
    print(f"Sample: {sample_id}; label={label}")
    print("PyTorch logits:", pytorch_logits.tolist())
    print("HLS float logits:", hls_logits.tolist())
    print("Maximum absolute error:", float(abs_error.max()))
    result = {
        "sample_id": sample_id,
        "label": label,
        "pytorch_logits": pytorch_logits.tolist(),
        "hls_float_logits": hls_logits.tolist(),
        "max_abs_error": float(abs_error.max()),
        "tolerance": args.atol,
        "status": "passed" if np.allclose(pytorch_logits, hls_logits, rtol=0.0, atol=args.atol) else "failed",
        "scope": "host-float equivalence; not fixed-point Vitis C simulation",
    }
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not np.allclose(pytorch_logits, hls_logits, rtol=0.0, atol=args.atol):
        raise SystemExit(f"FAILED: logits differ by more than atol={args.atol}")
    print("PASS: host-float HLS kernel matches PyTorch")


if __name__ == "__main__":
    main()
