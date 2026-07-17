"""Demo — Section 5 ruptures onset/offset detection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from neural_signal_toolkit import detection, viz

FS = 1000.0
t = np.arange(0, 3.0, 1 / FS)
# Two EMG-like bursts
x = 0.05 * np.random.default_rng(0).normal(size=t.size)
x[(t > 0.6) & (t < 1.1)] += 0.8 * np.random.default_rng(1).normal(size=np.sum((t > 0.6) & (t < 1.1)))
x[(t > 1.8) & (t < 2.4)] += 1.0 * np.random.default_rng(2).normal(size=np.sum((t > 1.8) & (t < 2.4)))

result = detection.detect_changepoints(x, FS, pen=8.0)
print("Breakpoints (idx):", result.breakpoints)
for seg in result.segments:
    print(f"Burst {seg.onset_s:.3f}s → {seg.offset_s:.3f}s")

fig, ax = plt.subplots(figsize=(11, 3))
viz.plot_bursts(x, FS, result.segments, ax=ax)
fig.tight_layout()
plt.show()
