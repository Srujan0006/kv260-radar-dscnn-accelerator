#!/usr/bin/env python3
"""Export trained Radar2DCNN to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/radar_2dcnn.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model.eval()
    dummy = torch.randn(1, 1, 100, 100)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["spectrogram"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX: {output}")

    try:
        import onnx
        model_onnx = onnx.load(str(output))
        onnx.checker.check_model(model_onnx)
        print("ONNX checker: OK")
    except Exception as exc:
        print(f"ONNX checker warning: {exc}")


if __name__ == "__main__":
    main()
