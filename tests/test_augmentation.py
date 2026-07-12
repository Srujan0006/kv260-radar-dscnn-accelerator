import numpy as np

from radar2dcnn.dataset import (
    CachedSpectrogramDataset,
    RadarArrays,
    RadarSpecAugment,
)


def test_augmentation_does_not_mutate_cached_image():
    images = np.full((1, 1, 100, 100), 0.5, dtype=np.float32)
    original = images.copy()
    dataset = CachedSpectrogramDataset(
        RadarArrays(X=images, y=np.asarray([0])), transform=RadarSpecAugment()
    )

    augmented, label = dataset[0]
    assert augmented.shape == (1, 100, 100)
    assert label == 0
    np.testing.assert_array_equal(images, original)


def test_dataset_without_transform_returns_cached_values():
    images = np.full((1, 1, 100, 100), 0.25, dtype=np.float32)
    dataset = CachedSpectrogramDataset(RadarArrays(X=images, y=np.asarray([2])))
    image, label = dataset[0]
    np.testing.assert_array_equal(image.numpy(), images[0])
    assert label == 2
