"""Synthetic physiological-like signals for examples and tests."""

from __future__ import annotations

import numpy as np


def make_emg_with_stim(
    fs: float = 2000.0,
    duration_s: float = 2.0,
    stim_hz: float = 30.0,
    stim_amp: float = 2.0,
    emg_amp: float = 0.4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(signal, clean_emg, stim_only)`` with impulsive stim artifacts.

    Stim is modeled as brief biphasic spikes at ``stim_hz`` plus a quiet
    artifact-dominant preamble useful for ACSR training demos.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Band-limited EMG-like noise
    emg = emg_amp * rng.normal(size=n)
    # Simple burst envelope in the second half
    burst = ((t > 0.8) & (t < 1.5)).astype(float)
    emg *= 0.15 + burst

    stim = np.zeros(n)
    period = int(round(fs / stim_hz))
    width = max(int(0.001 * fs), 1)
    for i in range(0, n - width, period):
        stim[i : i + width] = stim_amp
        stim[i + width : i + 2 * width] = -0.6 * stim_amp

    mixed = emg + stim
    return mixed, emg, stim


def make_artifact_reference(
    fs: float = 2000.0,
    duration_s: float = 3.0,
    stim_hz: float = 30.0,
    stim_amp: float = 2.0,
    seed: int = 1,
) -> np.ndarray:
    """Quiet standing-like reference: stim + small baseline noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    _, _, stim = make_emg_with_stim(
        fs=fs,
        duration_s=duration_s,
        stim_hz=stim_hz,
        stim_amp=stim_amp,
        emg_amp=0.05,
        seed=seed,
    )
    return stim + 0.02 * rng.normal(size=n)
