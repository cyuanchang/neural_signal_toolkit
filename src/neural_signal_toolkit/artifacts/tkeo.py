"""Teager–Kaiser Energy Operator (TKEO) utilities.

Discrete TKEO (vectorized), matching the common form used in the
MathWorks File Exchange vectorized TKEO implementation:

    psi[n] = x[n]^2 - x[n-1] * x[n+1]
"""

from __future__ import annotations

import numpy as np

from neural_signal_toolkit._utils import as_1d_float, zscore


def tkeo(x: np.ndarray) -> np.ndarray:
    """Compute the discrete Teager–Kaiser energy operator (length preserved).

    Endpoints use one-sided replication so the output length matches ``x``.
    """
    x = as_1d_float(x)
    if x.size < 3:
        raise ValueError("TKEO requires at least 3 samples")

    psi = np.empty_like(x)
    psi[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    psi[0] = psi[1]
    psi[-1] = psi[-2]
    return psi


def tkeo_zscore(x: np.ndarray) -> np.ndarray:
    """TKEO followed by z-score normalization."""
    return zscore(tkeo(x))
