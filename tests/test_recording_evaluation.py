import numpy as np

from radar2dcnn.evaluation import (
    aggregate_logits_by_group,
    aggregate_recording_predictions,
    detailed_classification_metrics,
)


def test_recording_logits_are_averaged_per_source_row():
    logits = np.asarray(
        [[4.0, 0.0], [0.0, 2.0], [0.0, 3.0], [0.0, 5.0]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1])
    groups = np.asarray([10, 10, 20, 20])

    group_ids, grouped_labels, grouped_logits = aggregate_logits_by_group(
        logits, labels, groups
    )
    np.testing.assert_array_equal(group_ids, [10, 20])
    np.testing.assert_array_equal(grouped_labels, [0, 1])
    np.testing.assert_allclose(grouped_logits, [[2.0, 1.0], [0.0, 4.0]])


def test_detailed_metrics_include_true_and_predicted_distributions():
    metrics = detailed_classification_metrics(
        np.asarray([0, 0, 0, 1]),
        np.asarray([0, 1, 1, 1]),
        ["Drone", "Bird"],
    )
    assert metrics["true_distribution"] == {"Drone": 3, "Bird": 1}
    assert metrics["predicted_distribution"] == {"Drone": 1, "Bird": 3}


def test_recording_aggregation_methods_can_disagree():
    # One very strong Drone window competes with two moderately confident Bird
    # windows. This exercises probability, vote, maximum, and top-fraction paths.
    logits = np.asarray([[4.0, 0.0], [0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    labels = np.asarray([1, 1, 1])
    groups = np.asarray([7, 7, 7])

    group_ids, grouped_labels, predictions = aggregate_recording_predictions(
        logits, labels, groups, top_fraction=0.30
    )
    np.testing.assert_array_equal(group_ids, [7])
    np.testing.assert_array_equal(grouped_labels, [1])
    assert predictions["mean_probability"].tolist() == [1]
    assert predictions["majority_vote"].tolist() == [1]
    assert predictions["median_probability"].tolist() == [1]
    assert predictions["max_class_confidence"].tolist() == [0]
    assert predictions["top_confidence_mean"].tolist() == [0]


def test_majority_vote_tie_uses_mean_probability():
    logits = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _, _, predictions = aggregate_recording_predictions(
        logits, np.asarray([0, 0]), np.asarray([3, 3])
    )
    assert predictions["majority_vote"].tolist() == [0]
