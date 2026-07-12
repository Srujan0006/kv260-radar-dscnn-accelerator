"""Grouped, stratified fold construction for recording-row validation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_nested_grouped_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create outer test folds with grouped inner validation splits.

    Every source group appears in the outer test set exactly once. Within each
    outer fold, train and validation are also disjoint by source group.
    """
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and groups must be one-dimensional and equally sized")
    if n_splits < 2 or inner_splits < 2:
        raise ValueError("n_splits and inner_splits must be at least 2")

    indices = np.arange(len(labels))
    group_ids, group_inverse = np.unique(groups, return_inverse=True)
    group_labels = np.empty(len(group_ids), dtype=np.int64)
    for group_index, group_id in enumerate(group_ids):
        unique_labels = np.unique(labels[group_inverse == group_index])
        if len(unique_labels) != 1:
            raise ValueError(
                f"Source group {group_id!r} contains labels {unique_labels.tolist()}"
            )
        group_labels[group_index] = unique_labels[0]

    class_group_counts = np.bincount(group_labels)
    if np.any(class_group_counts < n_splits):
        raise ValueError(
            f"Each class needs at least {n_splits} source groups for outer folds; "
            f"got {class_group_counts.tolist()}"
        )

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_index, (train_val_group_idx, test_group_idx) in enumerate(
        outer.split(group_ids, group_labels)
    ):
        inner_group_labels = group_labels[train_val_group_idx]
        inner_class_counts = np.bincount(inner_group_labels)
        if np.any(inner_class_counts < inner_splits):
            raise ValueError(
                f"Fold {fold_index + 1} needs at least {inner_splits} training "
                f"groups per class for its inner split; got {inner_class_counts.tolist()}"
            )
        inner = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed + fold_index + 1,
        )
        relative_train_group_idx, relative_val_group_idx = next(
            inner.split(train_val_group_idx, inner_group_labels)
        )
        train_groups = group_ids[train_val_group_idx[relative_train_group_idx]]
        val_groups = group_ids[train_val_group_idx[relative_val_group_idx]]
        test_groups = group_ids[test_group_idx]
        train_idx = indices[np.isin(groups, train_groups)]
        val_idx = indices[np.isin(groups, val_groups)]
        test_idx = indices[np.isin(groups, test_groups)]

        group_sets = [set(groups[idx].tolist()) for idx in (train_idx, val_idx, test_idx)]
        if not (
            group_sets[0].isdisjoint(group_sets[1])
            and group_sets[0].isdisjoint(group_sets[2])
            and group_sets[1].isdisjoint(group_sets[2])
        ):
            raise AssertionError(f"Group leakage in fold {fold_index + 1}")
        folds.append((train_idx, val_idx, test_idx))

    test_indices = np.concatenate([fold[2] for fold in folds])
    if len(test_indices) != len(labels) or not np.array_equal(
        np.sort(test_indices), indices
    ):
        raise AssertionError("Outer test folds do not partition every sample exactly once")
    return folds
