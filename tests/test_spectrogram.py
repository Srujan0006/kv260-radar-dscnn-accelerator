import numpy as np

from radar2dcnn.spectrogram import (
    iq_to_spectrogram,
    reshape_iq_segment,
    signal_to_spectrogram,
    topk_energy_mean_spectrogram,
)


def test_reshape_flat_complex():
    x = (np.random.randn(1280) + 1j * np.random.randn(1280)).astype(np.complex64)
    y = reshape_iq_segment(x)
    assert y.shape == (5, 256)


def test_spectrogram_shape_range():
    x = (np.random.randn(1280) + 1j * np.random.randn(1280)).astype(np.complex64)
    img = iq_to_spectrogram(x, image_size=100)
    assert img.shape == (100, 100)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0001


def test_topk_energy_mean_uses_three_strongest_bins():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(5, 256)) + 1j * rng.normal(size=(5, 256))
    scales = np.asarray([0.1, 4.0, 1.0, 3.0, 2.0])[:, None]
    iq = (base * scales).astype(np.complex64)

    actual = topk_energy_mean_spectrogram(iq, top_k=3, image_size=32)
    energy = np.mean(np.abs(iq) ** 2, axis=1)
    indices = np.argsort(energy)[-3:]
    expected = np.mean(
        [signal_to_spectrogram(iq[index], image_size=32) for index in indices],
        axis=0,
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual.shape == (32, 32)
    assert actual.dtype == np.float32


def test_topk_mode_supported_by_main_conversion_function():
    x = (np.random.randn(5, 256) + 1j * np.random.randn(5, 256)).astype(
        np.complex64
    )
    img = iq_to_spectrogram(x, image_size=24, range_mode="topk_energy_mean", top_k=3)
    assert img.shape == (24, 24)
