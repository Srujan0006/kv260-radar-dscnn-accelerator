#!/usr/bin/env python3
"""Validate ONNX structure and compare its logits with PyTorch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="outputs/real/radar_2dcnn_dpu_stride32.onnx")
    parser.add_argument(
        "--checkpoint",
        default="outputs/real/train_max_stride32_4class_dpu/best_model.pth",
    )
    parser.add_argument(
        "--cache", default="outputs/real/spectrogram_cache_max_stride32.npz"
    )
    parser.add_argument("--output", default="outputs/evidence/onnx_validation.json")
    args = parser.parse_args()

    model_proto = onnx.load(args.onnx)
    onnx.checker.check_model(model_proto)
    with np.load(args.cache, allow_pickle=False) as data:
        key = "images" if "images" in data.files else "X"
        sample = np.asarray(data[key][:1], dtype=np.float32)
    pytorch = load_checkpoint(args.checkpoint)
    with torch.no_grad():
        expected = pytorch(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    result = {
        "onnx_checker": "passed",
        "pytorch_onnx_max_abs_error": max_error,
        "input_shape": list(sample.shape),
        "output_shape": list(actual.shape),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if max_error > 1e-4:
        raise SystemExit("ONNX logits differ from PyTorch by more than 1e-4")


if __name__ == "__main__":
    main()
