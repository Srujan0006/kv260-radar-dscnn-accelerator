import numpy as np

from radar2dcnn.imbalance import compute_class_weights, make_weighted_sampler


def test_capped_weights_limit_extreme_minority_weight():
    counts = np.asarray([982, 61, 64, 8])
    inverse = compute_class_weights(counts, "inverse")
    capped = compute_class_weights(counts, "capped", max_weight=8.0)

    assert inverse is not None and capped is not None
    assert inverse[-1] > 30
    assert capped[-1] == 8.0
    np.testing.assert_allclose(capped[:-1], inverse[:-1])


def test_sqrt_and_effective_number_weights_are_finite():
    counts = np.asarray([100, 20, 10, 5])
    for strategy in ("sqrt", "effective_num"):
        weights = compute_class_weights(counts, strategy)
        assert weights is not None
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)


def test_sampler_only_uses_normal_loss_and_balanced_sample_weights():
    labels = np.asarray([0, 0, 0, 0, 1])
    counts = np.bincount(labels)
    assert compute_class_weights(counts, "sampler_only") is None
    sampler = make_weighted_sampler(labels, counts, seed=42)
    weights = np.asarray(sampler.weights)
    assert weights[-1] == 4 * weights[0]
