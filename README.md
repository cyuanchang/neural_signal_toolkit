# Neural Signal Toolkit

A practical Python toolkit for **physiological / neural signal processing**. It wraps battle-tested packages (`scipy`, `scikit-learn`, `ruptures`, `PyWavelets`) into a clear, sectioned API so you spend less time wiring pipelines and more time interpreting physiology.

**Focus areas**

1. Standardized filtering (high / low / band / notch)  
2. Stimulation artifact rejection (naive TKEO path + adaptive ACSR)  
3. Time–frequency / power spectrograms  
4. PCA · LDA · ICA  
5. Rupture-based onset / offset (burst) detection  

---

## Install

```bash
cd \neural_signal_toolkit
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from neural_signal_toolkit import filtering, artifacts, timefreq, detection
from neural_signal_toolkit.synth import make_emg_with_stim, make_artifact_reference

fs = 2000.0
x, emg, stim = make_emg_with_stim(fs=fs)
ref = make_artifact_reference(fs=fs)

# Bandpass
x_bp = filtering.bandpass(x, 20, 450, fs)

# Naive stim-artifact highlight (HPF - TKEO - z-score - threshold)
naive = artifacts.detect_artifacts_tkeo(x, fs, highpass_hz=100, threshold=3.0)

# 2b) Adaptive ACSR (Kim et al., 2021)
acsr = artifacts.acsr_filter(x, artifact_ref=ref, fs=fs)

# 3) Spectrogram
spec = timefreq.spectrogram(acsr.cleaned, fs)

# 5) Burst onset / offset via ruptures
bursts = detection.detect_onset_offset(acsr.cleaned, fs)
```

Runnable demos live in [`examples/`](examples/).

---

## Section 1 — Standard filtering

| Function | Role |
|---|---|
| `filtering.lowpass` | Attenuate above cutoff |
| `filtering.highpass` | Attenuate below cutoff |
| `filtering.bandpass` | Keep a band (e.g. EMG 4–150 Hz) |
| `filtering.bandstop` | Reject a band |
| `filtering.notch` / `multi_notch` | Narrow IIR notches (line noise / harmonics) eg 60Hz |

Defaults use Butterworth SOS + zero-phase `sosfiltfilt` / `filtfilt` so phase distortion does not shift burst timing.

```python
from neural_signal_toolkit import filtering
y = filtering.bandpass(x, low=20, high=450, fs=2000)
y = filtering.notch(y, freq=60, fs=2000, q=30)
```

---

## Section 2 — Artifact rejection (stimulation)

Electrical stimulation (FES, tSCS, …) injects large, brief contaminants into EMG/neural recordings. This toolkit exposes **two complementary solutions**.

### 2.1 Naive / fast path — HPF + TKEO + threshold

Inspired by practice + the vectorized Teager–Kaiser Energy Operator used in MATLAB  
(https://www.mathworks.com/matlabcentral/fileexchange/45406-teager-keiser-energy-operator-vectorized):

1. **High-pass @ 100 Hz** — emphasize sharp spikes, suppress slower physiological content  
2. **TKEO** — $\psi[n] = x[n]^2 - x[n-1]x[n+1]$ highlights instantaneous energy peaks  
3. **Z-score** normalize TKEO  
4. Compare against a **manually tuned threshold** and blank / interpolate flagged samples  

```python
from neural_signal_toolkit.artifacts import detect_artifacts_tkeo
result = detect_artifacts_tkeo(x, fs=2000, highpass_hz=100, threshold=3.0)
# result.mask, result.tkeo_z, result.cleaned
```

### 2.2 Adaptive path — ACSR (Kim et al., 2021)

**Artifact Component Specific Rejection** learns a frequency-domain artifact template from an *artifact-dominant* reference (e.g. quiet standing under stim), then subtracts that template from task windows while preserving phase.

Reference paper (also in `docs/references/`):

> Kim et al. (2021). *A Novel Technique to Reject Artifact Components for Surface EMG Signals Recorded During Walking With Transcutaneous Spinal Cord Stimulation.* Front. Hum. Neurosci. 15:660583.  
> https://doi.org/10.3389/fnhum.2021.660583  
> MATLAB code: https://github.com/mjkim0927/kim-frontiers-2021

**Algorithm (paper defaults):**

1. Segment reference with **200 ms** windows, **100 ms** overlap (~3 s training)  
2. FFT each window → take **max magnitude per frequency bin** → artifact parameters $Y_{\mathrm{artifact}}$  
3. For each analysis window: $|y'| = \max(|y| - Y_{\mathrm{artifact}},\, 0)$, keep $\angle y$, IFFT  

```python
from neural_signal_toolkit.artifacts import acsr_filter, ACSRParams

params = ACSRParams(window_ms=200, overlap_ms=100, train_ms=3000)
out = acsr_filter(task_emg, artifact_ref=standing_under_stim, fs=2000, params=params)
clean = out.cleaned
```
---

## Section 3 — Time–frequency / spectrograms

```python
from neural_signal_toolkit import timefreq
spec = timefreq.spectrogram(x, fs=2000)          # STFT PSD
f, t, Zxx = timefreq.stft_complex(x, fs=2000)
bp = timefreq.band_power(x, fs=2000, fmin=20, fmax=100)
freqs, t, power = timefreq.cwt_scalogram(x, fs=2000)  # optional CWT
```

---

## Section 4 — PCA · LDA · ICA

Friendly wrappers around `scikit-learn` for common neural-computation workflows:

```python
from neural_signal_toolkit import decomposition

pca = decomposition.fit_pca(X)                 # samples × features
lda = decomposition.fit_lda(X, y)              # supervised
ica = decomposition.fit_ica(X)
ica_ch = decomposition.ica_unmix_channels(C)   # channels × time
```

- **PCA** — variance / dimensionality reduction, visualization  
- **LDA** — class-separating projection (e.g. condition decoding)  
- **ICA** — blind source separation / artifact component hunting  

---

## Section 5 — Rupture-based onset / offset

Uses [`ruptures`](https://centre-borelli.github.io/ruptures-docs/) change-point detection on a rectified, smoothed envelope, then pairs high-activity segments as bursts:

```python
from neural_signal_toolkit import detection
result = detection.detect_changepoints(emg, fs=2000)
for seg in result.segments:
    print(seg.onset_s, seg.offset_s)
```

---

## Repository layout

```text
neural_signal_toolkit/
├── README.md
├── pyproject.toml
├── examples/
│   ├── 01_filtering_demo.py
│   ├── 02_artifact_rejection_demo.py
│   ├── 03_timefreq_demo.py
│   ├── 04_decomposition_demo.py
│   └── 05_onset_detection_demo.py
├── docs/references/          # ACSR paper PDF
├── tests/
└── src/neural_signal_toolkit/
    ├── filtering/
    ├── artifacts/            # TKEO + ACSR
    ├── timefreq/
    ├── decomposition/
    ├── detection/
    ├── viz/
    └── synth.py
```

---

## Tests

```bash
pytest -q
```

---

## Citation / references

If you use the ACSR path, please cite Kim et al. (2021). If you use TKEO, see the Teager–Kaiser literature and the MATLAB File Exchange vectorized implementation linked above.
