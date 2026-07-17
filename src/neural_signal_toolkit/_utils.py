"""Shared helpers for 1-D physiological signals."""

from __future__ import annotations

import numpy as np


def as_1d_float(signal: np.ndarray) -> np.ndarray:
    """Return a contiguous 1-D float64 array."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D signal, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def ensure_fs(fs: float) -> float:
    """Validate sampling rate (Hz)."""
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}")
    return fs


def time_vector(n_samples: int, fs: float) -> np.ndarray:
    """Return time axis in seconds for ``n_samples`` at sampling rate ``fs``."""
    return np.arange(n_samples, dtype=float) / ensure_fs(fs)


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardize to zero mean / unit variance."""
    x = as_1d_float(x)
    return (x - np.mean(x)) / (np.std(x) + eps)
