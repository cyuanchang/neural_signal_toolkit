"""Section 2 — stimulation / recording artifact rejection."""

from neural_signal_toolkit.artifacts.acsr import (
    ACSRParams,
    ACSRResult,
    acsr_filter,
    apply_acsr,
    estimate_artifact_spectrum,
)
from neural_signal_toolkit.artifacts.naive_tkeo import TKEOArtifactResult, detect_artifacts_tkeo
from neural_signal_toolkit.artifacts.tkeo import tkeo, tkeo_zscore

__all__ = [
    "tkeo",
    "tkeo_zscore",
    "detect_artifacts_tkeo",
    "TKEOArtifactResult",
    "ACSRParams",
    "ACSRResult",
    "acsr_filter",
    "apply_acsr",
    "estimate_artifact_spectrum",
]
