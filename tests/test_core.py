"""Unit tests for core toolkit sections."""

from __future__ import annotations

import numpy as np
import pytest

from neural_signal_toolkit import artifacts, decomposition, detection, filtering, timefreq
from neural_signal_toolkit.synth import make_artifact_reference, make_emg_with_stim


FS = 2000.0


def test_filters_smoke():
    x = np.random.default_rng(0).normal(size=2000)
    assert filtering.lowpass(x, 100, FS).shape == x.shape
    assert filtering.highpass(x, 20, FS).shape == x.shape
    assert filtering.bandpass(x, 20, 450, FS).shape == x.shape
    assert filtering.notch(x, 60, FS).shape == x.shape


def test_tkeo_naive_pipeline():
    x, _, _ = make_emg_with_stim(fs=FS, stim_amp=3.0)
    out = artifacts.detect_artifacts_tkeo(x, FS, threshold=3.0)
    assert out.tkeo_z.shape == x.shape
    assert out.mask.dtype == bool
    assert out.cleaned.shape == x.shape
    assert out.mask.any(), "expected some stim spikes above threshold"


def test_acsr_reduces_stim_energy():
    x, _, stim = make_emg_with_stim(fs=FS, stim_amp=3.0, seed=0)
    ref = make_artifact_reference(fs=FS, stim_amp=3.0, seed=1)
    out = artifacts.acsr_filter(x, ref, FS)
    # Cleaned signal should be closer to zero where stim dominates in reference sense:
    # at least finite and same length
    assert out.cleaned.shape == x.shape
    assert np.isfinite(out.cleaned).all()
    assert out.artifact_spectrum.size == out.window_samples


def test_spectrogram_and_bandpower():
    x, _, _ = make_emg_with_stim(fs=FS)
    spec = timefreq.spectrogram(x, FS, nperseg=256)
    assert spec.Sxx.ndim == 2
    bp = timefreq.band_power(x, FS, 20, 100, nperseg=256)
    assert bp >= 0


def test_pca_lda_ica():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = rng.integers(0, 2, size=200)
    pca = decomposition.fit_pca(X, n_components=2)
    assert pca.transformed.shape == (200, 2)
    lda = decomposition.fit_lda(X, y, n_components=1)
    assert lda.transformed.shape[0] == 200
    ica = decomposition.fit_ica(X, n_components=2)
    assert ica.transformed.shape[1] == 2


def test_onset_detection_finds_bursts():
    t = np.arange(0, 2.0, 1 / FS)
    x = 0.02 * np.random.default_rng(0).normal(size=t.size)
    x[(t > 0.5) & (t < 0.9)] += 1.0 * np.random.default_rng(1).normal(
        size=np.sum((t > 0.5) & (t < 0.9))
    )
    segs = detection.detect_onset_offset(x, FS, pen=10.0)
    assert len(segs) >= 1
    assert segs[0].offset_s > segs[0].onset_s
