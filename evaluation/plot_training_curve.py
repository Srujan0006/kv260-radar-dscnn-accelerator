#!/usr/bin/env python3
"""Plot accuracy and loss curves from either training history format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Radar2DCNN training")
    args = parser.parse_args()

    payload = json.loads(Path(args.history).read_text())
    history = payload.get("history", payload) if isinstance(payload, dict) else payload
    epochs = [row["epoch"] for row in history]
    train_acc_key = "train_accuracy" if "train_accuracy" in history[0] else "train_acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history[0] else "val_acc"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row[train_acc_key] for row in history], label="train")
    axes[0].plot(epochs, [row[val_acc_key] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(args.title)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
