"""Lightweight plotting helpers for toolkit demos."""

from __future__ import annotations

from typing import SEQUENCE

import numpy as np

from neural_signal_toolkit._utils import as_1d_float, ensure_fs, time_vector


def plot_signal(
    x: np.ndarray,
    fs: float,
    ax=None,
    label: str | None = None,
    title: str | None = None,
):
    """Plot a 1-D time series."""
    import matplotlib.pyplot as plt

    x = as_1d_float(x)
    t = time_vector(x.size, fs)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x, label=label, linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)
    if label:
        ax.legend(loc="upper right")
    return ax


def plot_spectrogram(result, ax=None, title: str = "Spectrogram", clim=None):
    """Plot a :class:`~neural_signal_toolkit.timefreq.SpectrogramResult`."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(
        result.t,
        result.f,
        10 * np.log10(result.Sxx + 1e-20),
        shading="auto",
        cmap="viridis",
    )
    if clim is not None:
        mesh.set_clim(clim)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Power (dB)")
    return ax


def plot_bursts(
    x: np.ndarray,
    fs: float,
    segments: SEQUENCE,
    ax=None,
    title: str = "Detected bursts",
):
    """Overlay onset/offset segments on a signal."""
    import matplotlib.pyplot as plt

    ax = plot_signal(x, fs, ax=ax, title=title)
    for seg in segments:
        ax.axvspan(seg.onset_s, seg.offset_s, color="C3", alpha=0.25)
        ax.axvline(seg.onset_s, color="C2", linestyle="--", linewidth=1)
        ax.axvline(seg.offset_s, color="C1", linestyle="--", linewidth=1)
    return ax


__all__ = ["plot_signal", "plot_spectrogram", "plot_bursts"]
