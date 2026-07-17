"""Demo — Section 3 spectrogram."""

from __future__ import annotations

import matplotlib.pyplot as plt

from neural_signal_toolkit import filtering, timefreq, viz
from neural_signal_toolkit.synth import make_emg_with_stim

FS = 2000.0
x, _, _ = make_emg_with_stim(fs=FS)
x = filtering.bandpass(x, 20, 450, FS)
spec = timefreq.spectrogram(x, FS, nperseg=512)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
viz.plot_signal(x, FS, ax=axes[0], title="Band-passed signal")
viz.plot_spectrogram(spec, ax=axes[1], title="Power spectrogram")
fig.tight_layout()
plt.show()
