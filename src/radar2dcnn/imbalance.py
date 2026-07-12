"""Class-imbalance strategies shared by record-wise training workflows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

IMBALANCE_STRATEGIES = (
    "inverse",
    "sqrt",
    "capped",
    "effective_num",
    "sampler_only",
    "none",
)


def compute_class_weights(
    counts: np.ndarray,
    strategy: str,
    max_weight: float = 8.0,
    effective_beta: float = 0.999,
) -> np.ndarray | None:
    """Return loss weights, or ``None`` for sampler-only/unweighted training."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError(f"Every class must have a positive count, got {counts.tolist()}")
    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(f"Unsupported imbalance strategy: {strategy!r}")

    inverse = counts.sum() / (len(counts) * counts)
    if strategy == "inverse":
        return inverse
    if strategy == "sqrt":
        return np.sqrt(inverse)
    if strategy == "capped":
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        return np.minimum(inverse, max_weight)
    if strategy == "effective_num":
        if not 0.0 <= effective_beta < 1.0:
            raise ValueError("effective_beta must be in [0, 1)")
        effective = (1.0 - effective_beta) / (1.0 - np.power(effective_beta, counts))
        return effective * (len(counts) / effective.sum())
    return None


def make_weighted_sampler(
    labels: np.ndarray,
    counts: np.ndarray,
    seed: int,
) -> WeightedRandomSampler:
    """Sample classes approximately uniformly while retaining all examples."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.float64)
    sample_weights = 1.0 / counts[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
