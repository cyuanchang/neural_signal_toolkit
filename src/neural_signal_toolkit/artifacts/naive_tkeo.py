"""Naive stimulation-artifact highlighting via HPF + TKEO + threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_signal_toolkit._utils import as_1d_float, ensure_fs
from neural_signal_toolkit.artifacts.tkeo import tkeo, tkeo_zscore
from neural_signal_toolkit.filtering import highpass


@dataclass
class TKEOArtifactResult:
    """Outputs of the naive TKEO artifact pipeline."""

    filtered: np.ndarray
    """Signal after high-pass (emphasizes sharp spikes)."""

    tkeo: np.ndarray
    """Raw Teager–Kaiser energy."""

    tkeo_z: np.ndarray
    """Z-scored TKEO."""

    mask: np.ndarray
    """Boolean mask where ``tkeo_z`` exceeds ``threshold`` (artifact candidates)."""

    threshold: float
    cleaned: np.ndarray
    """Signal with artifact samples blanked (linear interpolation)."""


def detect_artifacts_tkeo(
    x: np.ndarray,
    fs: float,
    highpass_hz: float = 100.0,
    threshold: float = 3.0,
    blank: bool = True,
    filter_order: int = 4,
) -> TKEOArtifactResult:
    """Naive stim-artifact detection used as a fast first-pass method.

    Pipeline
    1. High-pass at ``highpass_hz`` (default 100 Hz) to emphasize instantaneous
       spikes and attenuate lower-frequency physiological content.
    2. Compute Teager–Kaiser Energy Operator (TKEO).
    3. Z-score normalize TKEO scores.
    4. Flag samples where z-scored TKEO exceeds a manually tuned ``threshold``.
    5. Optionally blank flagged samples and interpolate.

    Notes
    Threshold is intentionally manual — tune on a few trials with known stim
    timing. For a more adaptive spectral approach, see :func:`acsr_filter`.
    """
    ensure_fs(fs)
    x = as_1d_float(x)

    filtered = highpass(x, cutoff=highpass_hz, fs=fs, order=filter_order)
    energy = tkeo(filtered)
    energy_z = tkeo_zscore(filtered)
    mask = energy_z > float(threshold)

    cleaned = x.copy()
    if blank and np.any(mask):
        cleaned = _blank_interpolate(cleaned, mask)

    return TKEOArtifactResult(
        filtered=filtered,
        tkeo=energy,
        tkeo_z=energy_z,
        mask=mask,
        threshold=float(threshold),
        cleaned=cleaned,
    )


def _blank_interpolate(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace True-mask samples by linear interpolation from neighbors."""
    y = x.copy()
    idx = np.arange(y.size)
    good = ~mask
    if not np.any(good):
        return np.zeros_like(y)
    y[mask] = np.interp(idx[mask], idx[good], y[good])
    return y
