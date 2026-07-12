import numpy as np

from radar2dcnn.cross_validation import make_nested_grouped_folds


def test_nested_grouped_folds_partition_recordings_without_leakage():
    # Six source rows per class is enough for five outer folds. Give rows
    # different window counts to verify splitting is based on rows, not windows.
    labels = []
    groups = []
    group_id = 0
    for class_id in range(4):
        for within_class in range(6):
            window_count = within_class + 1
            labels.extend([class_id] * window_count)
            groups.extend([group_id] * window_count)
            group_id += 1
    labels = np.asarray(labels)
    groups = np.asarray(groups)

    folds = make_nested_grouped_folds(labels, groups, n_splits=5, inner_splits=4)
    seen_test_groups = []
    for train_idx, val_idx, test_idx in folds:
        split_groups = [set(groups[idx].tolist()) for idx in (train_idx, val_idx, test_idx)]
        assert split_groups[0].isdisjoint(split_groups[1])
        assert split_groups[0].isdisjoint(split_groups[2])
        assert split_groups[1].isdisjoint(split_groups[2])
        for idx in (train_idx, val_idx, test_idx):
            assert set(labels[idx].tolist()) == {0, 1, 2, 3}
        seen_test_groups.extend(split_groups[2])

    assert sorted(seen_test_groups) == list(range(24))
