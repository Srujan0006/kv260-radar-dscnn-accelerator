"""Metric and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .labels import CLASS_NAMES


def compute_metrics(y_true: Iterable[int], y_pred: Iterable[int], class_names: Sequence[str] = CLASS_NAMES):
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": float(acc), "confusion_matrix": cm, "report": report}


def save_confusion_matrix(cm: np.ndarray, output_path: str | Path, class_names: Sequence[str] = CLASS_NAMES):
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
