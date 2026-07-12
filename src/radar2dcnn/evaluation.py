"""Detailed window- and recording-level evaluation helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

RECORDING_AGGREGATION_NAMES = {
    "mean_logits": "Mean logits",
    "mean_probability": "Mean softmax probability",
    "majority_vote": "Majority vote",
    "median_probability": "Median probability",
    "max_class_confidence": "Max class confidence",
    "top_confidence_mean": "Top-30% confidence average",
}


def label_distribution(labels: np.ndarray, class_names: Sequence[str]) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def detailed_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Return JSON-safe aggregate, per-class, and distribution metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(len(class_names))
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "true_distribution": label_distribution(y_true, class_names),
        "predicted_distribution": label_distribution(y_pred, class_names),
        "per_class": {
            name: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_logits_by_group(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average window logits per recording and return group IDs, labels, logits."""
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) != len(groups):
        raise ValueError("logits, labels, and groups must describe the same samples")

    group_ids = np.unique(groups)
    grouped_labels: list[int] = []
    grouped_logits: list[np.ndarray] = []
    for group_id in group_ids:
        mask = groups == group_id
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) != 1:
            raise ValueError(
                f"Recording group {group_id!r} contains labels {unique_labels.tolist()}"
            )
        grouped_labels.append(int(unique_labels[0]))
        grouped_logits.append(np.mean(logits[mask], axis=0))
    return (
        group_ids,
        np.asarray(grouped_labels, dtype=np.int64),
        np.stack(grouped_logits).astype(np.float32),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable NumPy softmax over classes."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def aggregate_recording_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    top_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Predict each recording with several window aggregation methods.

    Majority-vote ties are resolved using the tied class with the highest mean
    softmax probability. Top-confidence averaging selects the fraction of
    windows with the largest maximum softmax probability, then averages their
    complete probability vectors.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) != len(groups):
        raise ValueError("logits, labels, and groups must describe the same samples")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    probabilities = softmax(logits)
    group_ids = np.unique(groups)
    grouped_labels: list[int] = []
    predictions = {method: [] for method in RECORDING_AGGREGATION_NAMES}
    for group_id in group_ids:
        mask = groups == group_id
        group_logits = logits[mask]
        group_probabilities = probabilities[mask]
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) != 1:
            raise ValueError(
                f"Recording group {group_id!r} contains labels {unique_labels.tolist()}"
            )
        grouped_labels.append(int(unique_labels[0]))

        mean_probability = group_probabilities.mean(axis=0)
        predictions["mean_logits"].append(int(np.argmax(group_logits.mean(axis=0))))
        predictions["mean_probability"].append(int(np.argmax(mean_probability)))
        predictions["median_probability"].append(
            int(np.argmax(np.median(group_probabilities, axis=0)))
        )
        predictions["max_class_confidence"].append(
            int(np.argmax(group_probabilities.max(axis=0)))
        )

        window_predictions = np.argmax(group_probabilities, axis=1)
        vote_counts = np.bincount(
            window_predictions, minlength=group_probabilities.shape[1]
        )
        tied_classes = np.flatnonzero(vote_counts == vote_counts.max())
        tie_break = tied_classes[np.argmax(mean_probability[tied_classes])]
        predictions["majority_vote"].append(int(tie_break))

        confidence = group_probabilities.max(axis=1)
        keep_count = max(1, int(np.ceil(len(confidence) * top_fraction)))
        top_indices = np.argsort(confidence)[-keep_count:]
        predictions["top_confidence_mean"].append(
            int(np.argmax(group_probabilities[top_indices].mean(axis=0)))
        )

    return (
        group_ids,
        np.asarray(grouped_labels, dtype=np.int64),
        {
            method: np.asarray(method_predictions, dtype=np.int64)
            for method, method_predictions in predictions.items()
        },
    )
