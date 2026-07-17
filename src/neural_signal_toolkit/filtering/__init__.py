"""standardized FIR/IIR filtering wrappers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as sp_signal

from neural_signal_toolkit._utils import as_1d_float, ensure_fs

FilterKind = Literal["butter", "cheby1", "cheby2", "ellip", "bessel"]
Btype = Literal["lowpass", "highpass", "bandpass", "bandstop"]


def _design_sos(
    btype: Btype,
    cutoff: float | tuple[float, float],
    fs: float,
    order: int = 4,
    kind: FilterKind = "butter",
    rp: float = 1.0,
    rs: float = 40.0,
) -> np.ndarray:
    fs = ensure_fs(fs)
    nyq = 0.5 * fs

    if btype in ("lowpass", "highpass"):
        wn = float(cutoff) / nyq
        if not 0 < wn < 1:
            raise ValueError(f"Cutoff must be in (0, Nyquist={nyq}), got {cutoff}")
    else:
        low, high = cutoff  # type: ignore[misc]
        wn = [float(low) / nyq, float(high) / nyq]
        if not (0 < wn[0] < wn[1] < 1):
            raise ValueError(f"Invalid band edges {cutoff} for Nyquist={nyq}")

    kwargs: dict = {"N": order, "Wn": wn, "btype": btype, "analog": False, "output": "sos"}
    if kind == "butter":
        return sp_signal.butter(**kwargs)
    if kind == "cheby1":
        return sp_signal.cheby1(rp=rp, **kwargs)
    if kind == "cheby2":
        return sp_signal.cheby2(rs=rs, **kwargs)
    if kind == "ellip":
        return sp_signal.ellip(rp=rp, rs=rs, **kwargs)
    if kind == "bessel":
        return sp_signal.bessel(**kwargs)
    raise ValueError(f"Unknown filter kind: {kind}")


def apply_sos(x: np.ndarray, sos: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    """Apply second-order sections (forward-backward if ``zero_phase``)."""
    x = as_1d_float(x)
    if zero_phase:
        return sp_signal.sosfiltfilt(sos, x)
    return sp_signal.sosfilt(sos, x)


def lowpass(
    x: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    kind: FilterKind = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """Low-pass filter a 1-D signal."""
    sos = _design_sos("lowpass", cutoff, fs, order=order, kind=kind)
    return apply_sos(x, sos, zero_phase=zero_phase)


def highpass(
    x: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    kind: FilterKind = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """High-pass filter a 1-D signal."""
    sos = _design_sos("highpass", cutoff, fs, order=order, kind=kind)
    return apply_sos(x, sos, zero_phase=zero_phase)


def bandpass(
    x: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
    kind: FilterKind = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """Band-pass filter a 1-D signal between ``low`` and ``high`` Hz."""
    sos = _design_sos("bandpass", (low, high), fs, order=order, kind=kind)
    return apply_sos(x, sos, zero_phase=zero_phase)


def bandstop(
    x: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
    kind: FilterKind = "butter",
    zero_phase: bool = True,
) -> np.ndarray:
    """Band-stop (notch-band) filter between ``low`` and ``high`` Hz."""
    sos = _design_sos("bandstop", (low, high), fs, order=order, kind=kind)
    return apply_sos(x, sos, zero_phase=zero_phase)


def notch(
    x: np.ndarray,
    freq: float,
    fs: float,
    q: float = 30.0,
    zero_phase: bool = True,
) -> np.ndarray:
    """IIR notch at ``freq`` Hz (power-line / stimulation harmonic helper).

    Parameters
    q : Quality factor. Higher ``q`` → narrower notch.
    """
    ensure_fs(fs)
    x = as_1d_float(x)
    b, a = sp_signal.iirnotch(w0=freq, Q=q, fs=fs)
    if zero_phase:
        return sp_signal.filtfilt(b, a, x)
    return sp_signal.lfilter(b, a, x)


def multi_notch(
    x: np.ndarray,
    freqs: list[float] | np.ndarray,
    fs: float,
    q: float = 30.0,
    zero_phase: bool = True,
) -> np.ndarray:
    """Cascade notches at each frequency in ``freqs``."""
    y = as_1d_float(x)
    for f0 in freqs:
        y = notch(y, float(f0), fs, q=q, zero_phase=zero_phase)
    return y


__all__ = [
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "notch",
    "multi_notch",
    "apply_sos",
]
