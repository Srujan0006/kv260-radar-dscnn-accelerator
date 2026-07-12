#!/usr/bin/env python3
"""Run compiled .xmodel on KV260 using VART.

This script assumes the KV260 has a working Vitis AI runtime and DPU overlay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))

from radar2dcnn.dataset import load_spectrogram_cache
from radar2dcnn.labels import CLASS_NAMES


def get_child_subgraph_dpu(graph):
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_graphs = []
    for child in children:
        if child.has_attr("device") and child.get_attr("device").upper() == "DPU":
            dpu_graphs.append(child)
    return dpu_graphs


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", required=True, help="Compiled KV260 xmodel")
    parser.add_argument("--data", required=True, help="Cached spectrogram .npz")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    try:
        import vart
        import xir
    except Exception as exc:
        raise RuntimeError(
            "Could not import vart/xir. Run this on KV260 with Vitis AI runtime installed."
        ) from exc

    arrays = load_spectrogram_cache(args.data)
    graph = xir.Graph.deserialize(args.xmodel)
    subgraphs = get_child_subgraph_dpu(graph)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in xmodel. Was it compiled for DPU?")

    runner = vart.Runner.create_runner(subgraphs[0], "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()

    input_shape = tuple(input_tensors[0].dims)
    output_shape = tuple(output_tensors[0].dims)
    print(f"Input tensor shape : {input_shape}")
    print(f"Output tensor shape: {output_shape}")

    batch = input_shape[0]
    total = min(args.num_samples, len(arrays.y))
    correct = 0

    for start in range(0, total, batch):
        end = min(start + batch, total)
        n = end - start
        x = arrays.X[start:end].astype(np.float32)

        # Model trained as NCHW. Some DPU runners use NHWC. Handle both.
        input_data = np.zeros(input_shape, dtype=np.float32)
        if len(input_shape) == 4 and input_shape[1] == 1:
            input_data[:n] = x
        elif len(input_shape) == 4 and input_shape[-1] == 1:
            input_data[:n] = np.transpose(x, (0, 2, 3, 1))
        else:
            raise RuntimeError(f"Unsupported DPU input shape: {input_shape}")

        output_data = np.empty(output_shape, dtype=np.float32)
        job_id = runner.execute_async([input_data], [output_data])
        runner.wait(job_id)

        logits = output_data[:n]
        if logits.ndim > 2:
            logits = logits.reshape(n, -1)
        probs = softmax_np(logits)
        pred = np.argmax(probs, axis=1)

        for i in range(n):
            true_label = int(arrays.y[start + i])
            pred_label = int(pred[i])
            correct += int(pred_label == true_label)
            print(
                f"sample={start+i:04d} true={CLASS_NAMES[true_label]:16s} "
                f"pred={CLASS_NAMES[pred_label]:16s} conf={float(probs[i, pred_label]):.3f}"
            )

    print(f"KV260 DPU demo accuracy on {total} samples: {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
