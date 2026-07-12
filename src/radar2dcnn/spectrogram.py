"""IQ to 100x100 micro-Doppler spectrogram conversion."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage, signal

RangeMode = Literal["center", "mean", "max", "topk_energy_mean"]


def reshape_iq_segment(iq: np.ndarray) -> np.ndarray:
    """Return IQ segment as complex array with shape [5, 256].

    Accepted inputs:
    - [1280] complex
    - [5, 256] complex
    - [1280, 2] real/imag
    - [5, 256, 2] real/imag
    """
    arr = np.asarray(iq)

    if np.iscomplexobj(arr):
        if arr.shape == (5, 256):
            return arr.astype(np.complex64)
        if arr.size == 1280:
            return arr.reshape(5, 256).astype(np.complex64)

    if arr.ndim >= 2 and arr.shape[-1] == 2:
        complex_arr = arr[..., 0] + 1j * arr[..., 1]
        if complex_arr.shape == (5, 256):
            return complex_arr.astype(np.complex64)
        if complex_arr.size == 1280:
            return complex_arr.reshape(5, 256).astype(np.complex64)

    raise ValueError(
        f"Unsupported IQ segment shape {arr.shape}. Expected [1280], [5,256], [1280,2], or [5,256,2]."
    )


def select_range_signal(iq_5x256: np.ndarray, mode: RangeMode = "center") -> np.ndarray:
    """Select or combine range cells into one 256-sample complex Doppler signal."""
    if mode == "center":
        return iq_5x256[2]
    if mode == "mean":
        return iq_5x256.mean(axis=0)
    if mode == "max":
        energy = np.mean(np.abs(iq_5x256) ** 2, axis=1)
        return iq_5x256[int(np.argmax(energy))]
    raise ValueError(f"Unsupported range_mode={mode!r}")


def normalize_image(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize spectrogram image to float32 range [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    image = image - np.nanmin(image)
    denom = np.nanmax(image) + eps
    return (image / denom).astype(np.float32)


def resize_to_square(image: np.ndarray, image_size: int = 100) -> np.ndarray:
    """Resize 2D array to image_size x image_size using bilinear interpolation."""
    zoom_y = image_size / image.shape[0]
    zoom_x = image_size / image.shape[1]
    resized = ndimage.zoom(image, (zoom_y, zoom_x), order=1)
    # Crop or pad defensively to exact size.
    out = np.zeros((image_size, image_size), dtype=np.float32)
    h = min(image_size, resized.shape[0])
    w = min(image_size, resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    return out


def signal_to_spectrogram(
    x: np.ndarray,
    image_size: int = 100,
    nperseg: int = 64,
    noverlap: int = 56,
    log_scale: bool = True,
) -> np.ndarray:
    """Convert one 1-D complex Doppler signal to a normalized image."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected a 1-D Doppler signal, got {x.shape}")
    if not np.iscomplexobj(x):
        raise ValueError("Doppler signal must contain complex IQ samples")

    # Remove DC component; micro-Doppler is carried by variations.
    x = x - np.mean(x)

    _, _, zxx = signal.stft(
        x,
        fs=1.0,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(noverlap, nperseg - 1),
        nfft=128,
        boundary=None,
        padded=False,
        return_onesided=False,
    )

    spec = np.abs(np.fft.fftshift(zxx, axes=0)) ** 2
    if log_scale:
        spec = 10.0 * np.log10(spec + 1e-12)

    spec = resize_to_square(spec, image_size=image_size)
    return normalize_image(spec).astype(np.float32)


def topk_energy_mean_spectrogram(
    iq_bins: np.ndarray,
    top_k: int = 3,
    image_size: int = 100,
    nperseg: int = 64,
    noverlap: int = 56,
    log_scale: bool = True,
) -> np.ndarray:
    """Average normalized spectrograms from the top-k energy range bins.

    ``iq_bins`` must have shape ``[num_range_bins, num_samples]``. Selection is
    based only on the current window, and no validation/test statistics are
    involved.
    """
    iq_bins = np.asarray(iq_bins)
    if iq_bins.ndim != 2 or not np.iscomplexobj(iq_bins):
        raise ValueError(
            "Expected complex IQ range bins [num_range_bins, num_samples], "
            f"got {iq_bins.shape}"
        )
    if top_k <= 0 or top_k > iq_bins.shape[0]:
        raise ValueError(
            f"top_k must be in [1, {iq_bins.shape[0]}], got {top_k}"
        )

    energy = np.mean(np.abs(iq_bins) ** 2, axis=1)
    indices = np.argsort(energy)[-top_k:]
    images = [
        signal_to_spectrogram(
            iq_bins[index],
            image_size=image_size,
            nperseg=nperseg,
            noverlap=noverlap,
            log_scale=log_scale,
        )
        for index in indices
    ]
    return np.mean(images, axis=0, dtype=np.float32).astype(np.float32)


def iq_to_spectrogram(
    iq: np.ndarray,
    image_size: int = 100,
    range_mode: RangeMode = "center",
    nperseg: int = 64,
    noverlap: int = 56,
    log_scale: bool = True,
    top_k: int = 3,
) -> np.ndarray:
    """Convert one radar IQ segment into a normalized 100x100 spectrogram.

    Returns:
        float32 array [image_size, image_size] in [0, 1].
    """
    iq_5x256 = reshape_iq_segment(iq)
    if range_mode == "topk_energy_mean":
        return topk_energy_mean_spectrogram(
            iq_5x256,
            top_k=top_k,
            image_size=image_size,
            nperseg=nperseg,
            noverlap=noverlap,
            log_scale=log_scale,
        )
    x = select_range_signal(iq_5x256, range_mode)
    return signal_to_spectrogram(
        x,
        image_size=image_size,
        nperseg=nperseg,
        noverlap=noverlap,
        log_scale=log_scale,
    )
