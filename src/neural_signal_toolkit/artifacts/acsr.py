"""Artifact Component Specific Rejection (ACSR) for stimulation artifacts.

Faithful Python replication of the method described in:
Kim et al. (2021). A Novel Technique to Reject Artifact Components for Surface EMG Signals Recorded During Walking With Transcutaneous Spinal
Cord Stimulation. Front. Hum. Neurosci. 15:660583. https://doi.org/10.3389/fnhum.2021.660583

Official published MATLAB implementation: https://github.com/mjkim0927/kim-frontiers-2021

Pipeline:
1. Train on an *artifact-dominant* segment (e.g., quiet standing under stim).
2. For each overlapping window, take the FFT magnitude spectrum.
3. Artifact template = **max** magnitude at each frequency bin across windows.
4. For analysis windows, subtract that template from the magnitude spectrum
   (floor at 0), keep original phase, and IFFT reconstruct.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import fft, ifft

from neural_signal_toolkit._utils import as_1d_float, ensure_fs


@dataclass
class ACSRParams:
    """ACSR hyperparameters (paper defaults for walking sEMG @ ~2 kHz)."""

    window_ms: float = 200.0
    overlap_ms: float = 100.0
    train_ms: float | None = None
    """If set, only the first ``train_ms`` of ``artifact_ref`` is used."""


@dataclass
class ACSRResult:
    cleaned: np.ndarray
    artifact_spectrum: np.ndarray
    """Max-magnitude artifact template (one-sided RFFT length)."""

    freqs: np.ndarray
    window_samples: int
    hop_samples: int


def _window_indices(n: int, win: int, hop: int) -> list[tuple[int, int]]:
    if n < win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def estimate_artifact_spectrum(
    artifact_ref: np.ndarray,
    fs: float,
    params: ACSRParams | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Learn ACSR artifact parameters from an artifact-dominant reference.
    Returns: artifact_spectrum, freqs, window_samples, hop_samples
    """
    params = params or ACSRParams()
    fs = ensure_fs(fs)
    ref = as_1d_float(artifact_ref)
    if params.train_ms is not None:
        n_train = int(round(params.train_ms * 1e-3 * fs))
        ref = ref[: max(n_train, 1)]

    win = max(int(round(params.window_ms * 1e-3 * fs)), 8)
    hop = max(int(round((params.window_ms - params.overlap_ms) * 1e-3 * fs)), 1)
    # Paper: overlap t = 100 ms with N = 200 ms → hop = 100 ms
    if params.overlap_ms >= params.window_ms:
        raise ValueError("overlap_ms must be < window_ms")

    spectra = []
    for start, end in _window_indices(ref.size, win, hop):
        seg = ref[start:end]
        if seg.size < win:
            seg = np.pad(seg, (0, win - seg.size))
        # Use full complex FFT magnitudes for one-to-one port of paper eqs.
        mag = np.abs(fft(seg, n=win))
        spectra.append(mag)

    artifact = np.max(np.stack(spectra, axis=0), axis=0)
    freqs = np.fft.fftfreq(win, d=1.0 / fs)
    return artifact, freqs, win, hop


def apply_acsr(
    x: np.ndarray,
    artifact_spectrum: np.ndarray,
    fs: float,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Apply a precomputed ACSR artifact template to signal ``x``."""
    x = as_1d_float(x)
    ensure_fs(fs)
    win = int(window_samples)
    hop = int(hop_samples)
    if artifact_spectrum.size != win:
        raise ValueError("artifact_spectrum length must equal window_samples")

    out = np.zeros_like(x)
    weight = np.zeros_like(x)
    taper = np.hanning(win) if win > 1 else np.ones(1)

    for start, end in _window_indices(x.size, win, hop):
        seg = x[start:end]
        pad = 0
        if seg.size < win:
            pad = win - seg.size
            seg = np.pad(seg, (0, pad))

        Y = fft(seg, n=win)
        mag = np.abs(Y)
        phase = np.angle(Y)
        mag_clean = np.maximum(mag - artifact_spectrum, 0.0)
        Y_clean = mag_clean * np.exp(1j * phase)
        seg_clean = np.real(ifft(Y_clean, n=win))

        use = win - pad
        out[start : start + use] += seg_clean[:use] * taper[:use]
        weight[start : start + use] += taper[:use]

    weight = np.maximum(weight, 1e-12)
    return out / weight


def acsr_filter(
    x: np.ndarray,
    artifact_ref: np.ndarray,
    fs: float,
    params: ACSRParams | None = None,
) -> ACSRResult:
    """End-to-end ACSR: train on ``artifact_ref``, clean ``x``.

    - ``artifact_ref``: quiet standing (or rest) under stimulation
    - ``x``: walking / task segment with the same stim settings
    - defaults: 200 ms windows, 100 ms overlap (Kim et al., 2021)
    """
    params = params or ACSRParams()
    artifact, freqs, win, hop = estimate_artifact_spectrum(artifact_ref, fs, params)
    cleaned = apply_acsr(x, artifact, fs, win, hop)
    return ACSRResult(
        cleaned=cleaned,
        artifact_spectrum=artifact,
        freqs=freqs,
        window_samples=win,
        hop_samples=hop,
    )


__all__ = [
    "ACSRParams",
    "ACSRResult",
    "estimate_artifact_spectrum",
    "apply_acsr",
    "acsr_filter",
]
