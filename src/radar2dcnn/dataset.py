"""Dataset loading utilities for radar IQ and cached spectrogram files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .labels import CLASS_NAMES, map_labels
from .spectrogram import iq_to_spectrogram

SIGNAL_KEYS = ["X", "iq", "data", "signals", "samples"]
LABEL_KEYS = ["y", "labels", "target", "targets", "classes"]
IMAGE_KEYS = ["images", "spectrograms", "X_img", "X"]


@dataclass
class RadarArrays:
    X: np.ndarray
    y: np.ndarray
    class_names: Tuple[str, ...] = tuple(CLASS_NAMES)


def _load_numpy_file(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        return {key: obj[key] for key in obj.files}

    # .npy case: could be plain array or dict-like object.
    arr = obj
    if isinstance(arr, np.ndarray) and arr.shape == ():
        item = arr.item()
        if isinstance(item, dict):
            return item
    if isinstance(arr, np.ndarray):
        return {"X": arr}

    raise ValueError(f"Unsupported NumPy file: {path}")


def _find_key(data: Dict[str, np.ndarray], candidates: Iterable[str]) -> Optional[str]:
    lower_to_real = {key.lower(): key for key in data.keys()}
    for cand in candidates:
        if cand in data:
            return cand
        if cand.lower() in lower_to_real:
            return lower_to_real[cand.lower()]
    return None


def load_radar_iq_npz(path: str | Path) -> RadarArrays:
    """Load raw IQ dataset from npz/npy.

    Returns X as original IQ array and y as int superclass labels.
    """
    data = _load_numpy_file(path)
    x_key = _find_key(data, SIGNAL_KEYS)
    y_key = _find_key(data, LABEL_KEYS)

    if x_key is None:
        raise KeyError(f"Could not find signal key. Available keys: {list(data.keys())}")
    if y_key is None:
        raise KeyError(f"Could not find label key. Available keys: {list(data.keys())}")

    X = data[x_key]
    y_raw = data[y_key]

    # If already integer 0..3, keep; otherwise map strings/detailed labels.
    try:
        y = np.asarray(y_raw, dtype=np.int64)
        if not np.all((0 <= y) & (y < len(CLASS_NAMES))):
            y = np.asarray(map_labels(y_raw), dtype=np.int64)
    except Exception:
        y = np.asarray(map_labels(y_raw), dtype=np.int64)

    if len(X) != len(y):
        raise ValueError(f"X length {len(X)} does not match y length {len(y)}")

    return RadarArrays(X=X, y=y)


def load_spectrogram_cache(path: str | Path) -> RadarArrays:
    """Load cached spectrogram dataset."""
    data = _load_numpy_file(path)
    x_key = _find_key(data, IMAGE_KEYS)
    y_key = _find_key(data, LABEL_KEYS)
    if x_key is None or y_key is None:
        raise KeyError(f"Expected image and label keys. Available keys: {list(data.keys())}")
    X = np.asarray(data[x_key], dtype=np.float32)
    y = np.asarray(data[y_key], dtype=np.int64)

    if X.ndim == 3:
        X = X[:, None, :, :]
    elif X.ndim == 4 and X.shape[-1] == 1:
        X = np.transpose(X, (0, 3, 1, 2))
    elif X.ndim == 4 and X.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Expected cached images [N,H,W] or [N,1,H,W], got {X.shape}")

    return RadarArrays(X=X, y=y)


class CachedSpectrogramDataset(Dataset):
    """PyTorch Dataset for cached spectrogram arrays."""

    def __init__(
        self,
        arrays: RadarArrays,
        indices: Optional[np.ndarray] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.X = arrays.X
        self.y = arrays.y
        self.indices = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        x = torch.from_numpy(self.X[real_idx]).float()
        if self.transform is not None:
            x = self.transform(x)
        y = int(self.y[real_idx])
        return x, y


class RadarSpecAugment:
    """Lightweight train-time augmentation for one-channel spectrograms."""

    def __init__(
        self,
        time_mask: int = 10,
        freq_mask: int = 8,
        noise_std: float = 0.02,
        max_time_shift: int = 5,
        max_freq_shift: int = 4,
        crop_fraction: float = 0.05,
    ):
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.noise_std = noise_std
        self.max_time_shift = max_time_shift
        self.max_freq_shift = max_freq_shift
        self.crop_fraction = crop_fraction

    @staticmethod
    def _mask(x: torch.Tensor, axis: int, max_width: int) -> None:
        size = x.shape[axis]
        width = random.randint(1, min(max_width, size))
        start = random.randint(0, size - width)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(start, start + width)
        x[tuple(slices)] = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected spectrogram [1,H,W], got {tuple(x.shape)}")

        # Clone before in-place masks so the cached NumPy image is never changed.
        x = x.clone() * random.uniform(0.85, 1.15)
        if self.noise_std > 0 and random.random() < 0.5:
            x.add_(torch.randn_like(x) * self.noise_std)

        if self.max_time_shift > 0 and random.random() < 0.5:
            shift = random.randint(-self.max_time_shift, self.max_time_shift)
            x = torch.roll(x, shifts=shift, dims=-1)
        if self.max_freq_shift > 0 and random.random() < 0.5:
            shift = random.randint(-self.max_freq_shift, self.max_freq_shift)
            x = torch.roll(x, shifts=shift, dims=-2)

        if self.time_mask > 0 and random.random() < 0.5:
            self._mask(x, axis=-1, max_width=self.time_mask)
        if self.freq_mask > 0 and random.random() < 0.5:
            self._mask(x, axis=-2, max_width=self.freq_mask)

        if self.crop_fraction > 0 and random.random() < 0.25:
            height, width = x.shape[-2:]
            crop_h = random.randint(0, max(1, int(height * self.crop_fraction)))
            crop_w = random.randint(0, max(1, int(width * self.crop_fraction)))
            cropped = x[:, crop_h : height - crop_h, crop_w : width - crop_w]
            x = F.interpolate(
                cropped.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return x.clamp_(0.0, 1.0)


class OnTheFlySpectrogramDataset(Dataset):
    """Dataset that computes spectrograms from IQ on demand.

    Slower than cache, but useful for debugging.
    """

    def __init__(
        self,
        arrays: RadarArrays,
        indices: Optional[np.ndarray] = None,
        image_size: int = 100,
        range_mode: str = "center",
    ):
        self.X = arrays.X
        self.y = arrays.y
        self.indices = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        self.image_size = image_size
        self.range_mode = range_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img = iq_to_spectrogram(self.X[real_idx], self.image_size, self.range_mode)
        x = torch.from_numpy(img[None, :, :]).float()
        y = int(self.y[real_idx])
        return x, y


def make_splits(
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/val/test splits."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=y,
    )
    temp_y = y[temp_idx]
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_fraction_of_temp,
        random_state=seed,
        stratify=temp_y,
    )
    return train_idx, val_idx, test_idx
