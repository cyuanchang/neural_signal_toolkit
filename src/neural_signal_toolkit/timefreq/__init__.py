"""Section 3 — time–frequency / power spectrogram transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

from neural_signal_toolkit._utils import as_1d_float, ensure_fs


@dataclass
class SpectrogramResult:
    """Power spectrogram container."""

    f: np.ndarray
    """Frequency bins (Hz)."""

    t: np.ndarray
    """Time bins (s)."""

    Sxx: np.ndarray
    """Power spectral density (V^2/Hz) or magnitude-squared, shape (F, T)."""

    method: str


def spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    window: str = "hann",
    scaling: str = "density",
    mode: str = "psd",
) -> SpectrogramResult:
    """STFT-based power spectrogram (SciPy wrapper with sensible neural defaults).

    Defaults choose ``nperseg`` ≈ 0.25 s of data when unspecified.
    """
    x = as_1d_float(x)
    fs = ensure_fs(fs)
    if nperseg is None:
        nperseg = max(int(round(0.25 * fs)), 16)
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = sp_signal.spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        mode=mode,
    )
    return SpectrogramResult(f=f, t=t, Sxx=Sxx, method="stft")


def stft_complex(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex STFT ``(f, t, Zxx)``."""
    x = as_1d_float(x)
    fs = ensure_fs(fs)
    if nperseg is None:
        nperseg = max(int(round(0.25 * fs)), 16)
    if noverlap is None:
        noverlap = nperseg // 2
    return sp_signal.stft(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )


def band_power(
    x: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    nperseg: int | None = None,
) -> float:
    """Mean PSD power in ``[fmin, fmax]`` via Welch."""
    x = as_1d_float(x)
    fs = ensure_fs(fs)
    if nperseg is None:
        nperseg = min(x.size, max(int(round(0.5 * fs)), 16))
    f, pxx = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        raise ValueError(f"No frequency bins in [{fmin}, {fmax}]")
    integrator = getattr(np, "trapezoid", None) or np.trapz
    return float(integrator(pxx[band], f[band]))


def cwt_scalogram(
    x: np.ndarray,
    fs: float,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous wavelet transform magnitude scalogram via PyWavelets.

    Returns ``(freqs_hz, time_s, power)`` with ``power = |CWT|^2``.
    """
    try:
        import pywt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pywavelets to use cwt_scalogram: pip install PyWavelets") from exc

    x = as_1d_float(x)
    fs = ensure_fs(fs)
    if scales is None:
        scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    t = np.arange(x.size) / fs
    return freqs, t, power


__all__ = [
    "SpectrogramResult",
    "spectrogram",
    "stft_complex",
    "band_power",
    "cwt_scalogram",
]
