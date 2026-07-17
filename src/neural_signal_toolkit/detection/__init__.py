"""rupture-based onset / offset (change-point) detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neural_signal_toolkit._utils import as_1d_float, ensure_fs


@dataclass
class BurstSegment:
    """One detected contiguous activation / burst interval."""

    onset_idx: int
    offset_idx: int
    onset_s: float
    offset_s: float


@dataclass
class ChangePointResult:
    """Ruptures change-point output plus optional burst pairing."""

    breakpoints: list[int]
    """Indices of detected change points (end-exclusive segment boundaries)."""

    segments: list[BurstSegment]
    signal: np.ndarray
    score: np.ndarray | None = None


def _to_envelope(x: np.ndarray, fs: float, smooth_ms: float = 20.0) -> np.ndarray:
    """Rectify + moving-average envelope for more stable change detection."""
    from scipy.ndimage import uniform_filter1d

    env = np.abs(as_1d_float(x))
    w = max(int(round(smooth_ms * 1e-3 * fs)), 1)
    return uniform_filter1d(env, size=w, mode="nearest")


def detect_changepoints(
    x: np.ndarray,
    fs: float,
    model: Literal["l2", "l1", "rbf", "normal"] = "l2",
    pen: float | None = None,
    n_bkps: int | None = None,
    min_size: int = 2,
    use_envelope: bool = True,
    smooth_ms: float = 20.0,
) -> ChangePointResult:
    """Detect change points with the ``ruptures`` package (PELT or Dynp).

    Parameters
    pen :
        Penalty for PELT. If both ``pen`` and ``n_bkps`` are None, uses a
        simple MAD-based heuristic.
    n_bkps :
        If set, use dynamic programming with a fixed number of breakpoints.
    use_envelope :
        Detect on a rectified smoothed envelope (typical for EMG bursts).
    """
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install ruptures: pip install ruptures") from exc

    fs = ensure_fs(fs)
    raw = as_1d_float(x)
    score = _to_envelope(raw, fs, smooth_ms=smooth_ms) if use_envelope else raw
    signal_2d = score.reshape(-1, 1)

    if n_bkps is not None:
        algo = rpt.Dynp(model=model, min_size=min_size).fit(signal_2d)
        bkps = algo.predict(n_bkps=n_bkps)
    else:
        if pen is None:
            mad = np.median(np.abs(score - np.median(score))) + 1e-12
            pen = 3.0 * mad**2 * np.log(len(score))
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal_2d)
        bkps = algo.predict(pen=pen)

    # ruptures always includes the signal end as the last breakpoint
    breakpoints = [int(b) for b in bkps[:-1]]
    segments = _pair_bursts(score, breakpoints, fs)
    return ChangePointResult(
        breakpoints=breakpoints,
        segments=segments,
        signal=score,
        score=score,
    )


def _pair_bursts(
    score: np.ndarray,
    breakpoints: list[int],
    fs: float,
) -> list[BurstSegment]:
    """Pair consecutive change points into high-activity burst segments.

    Segments whose mean envelope exceeds the global median are kept as bursts.
    """
    edges = [0, *breakpoints, len(score)]
    thr = float(np.median(score))
    bursts: list[BurstSegment] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        if float(np.mean(score[a:b])) > thr:
            bursts.append(
                BurstSegment(
                    onset_idx=a,
                    offset_idx=b - 1,
                    onset_s=a / fs,
                    offset_s=(b - 1) / fs,
                )
            )
    return bursts


def detect_onset_offset(
    x: np.ndarray,
    fs: float,
    **kwargs,
) -> list[BurstSegment]:
    """Convenience wrapper returning only burst onset/offset segments."""
    return detect_changepoints(x, fs, **kwargs).segments


__all__ = [
    "BurstSegment",
    "ChangePointResult",
    "detect_changepoints",
    "detect_onset_offset",
]
