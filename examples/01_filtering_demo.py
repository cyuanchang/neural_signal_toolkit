"""Demo — Section 1 standard filtering on a synthetic EMG+stim mixture."""

from __future__ import annotations

import matplotlib.pyplot as plt

from neural_signal_toolkit import filtering, viz
from neural_signal_toolkit.synth import make_emg_with_stim

FS = 2000.0
x, _, _ = make_emg_with_stim(fs=FS)

x_hp = filtering.highpass(x, 100, FS)
x_bp = filtering.bandpass(x, 20, 450, FS)
x_notch = filtering.notch(x_bp, 60, FS, q=30)

fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
viz.plot_signal(x, FS, ax=axes[0], title="Raw mixture")
viz.plot_signal(x_hp, FS, ax=axes[1], title="High-pass 100 Hz")
viz.plot_signal(x_bp, FS, ax=axes[2], title="Band-pass 20–450 Hz")
viz.plot_signal(x_notch, FS, ax=axes[3], title="Band-pass + 60 Hz notch")
fig.tight_layout()
plt.show()
