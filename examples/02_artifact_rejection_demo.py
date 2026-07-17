"""Demo — Section 2 naive TKEO vs adaptive ACSR artifact rejection."""

from __future__ import annotations

import matplotlib.pyplot as plt

from neural_signal_toolkit import artifacts, viz
from neural_signal_toolkit.synth import make_artifact_reference, make_emg_with_stim

FS = 2000.0
x, emg, _ = make_emg_with_stim(fs=FS, stim_amp=2.5)
ref = make_artifact_reference(fs=FS, stim_amp=2.5)

naive = artifacts.detect_artifacts_tkeo(x, FS, highpass_hz=100, threshold=3.0)
acsr = artifacts.acsr_filter(x, artifact_ref=ref, fs=FS)

fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
viz.plot_signal(x, FS, ax=axes[0], title="Contaminated signal")
viz.plot_signal(naive.tkeo_z, FS, ax=axes[1], title=f"Z-scored TKEO (thr={naive.threshold})")
axes[1].axhline(naive.threshold, color="r", ls="--")
viz.plot_signal(naive.cleaned, FS, ax=axes[2], title="Naive TKEO-cleaned")
viz.plot_signal(acsr.cleaned, FS, ax=axes[3], title="ACSR-cleaned (Kim et al. 2021)")
viz.plot_signal(emg, FS, ax=axes[4], title="Ground-truth clean EMG (synthetic)")
fig.tight_layout()
plt.show()
