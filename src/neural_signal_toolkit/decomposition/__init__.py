"""Section 4 — standard neural / multivariate decompositions (PCA, LDA, ICA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass
class FitTransformResult:
    """Generic container for fitted transformer + projected data."""

    model: object
    transformed: np.ndarray
    components: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None


def fit_pca(
    X: np.ndarray,
    n_components: int | float | None = None,
    whiten: bool = False,
    random_state: int = 0,
) -> FitTransformResult:
    """Principal Component Analysis on samples × features matrix ``X``."""
    X = np.asarray(X, dtype=float)
    model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    Z = model.fit_transform(X)
    return FitTransformResult(
        model=model,
        transformed=Z,
        components=model.components_,
        explained_variance_ratio=model.explained_variance_ratio_,
    )


def fit_lda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int | None = None,
    solver: str = "svd",
) -> FitTransformResult:
    """Linear Discriminant Analysis (supervised projection)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    model = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
    Z = model.fit_transform(X, y)
    components = getattr(model, "scalings_", None)
    if components is not None:
        components = components.T
    return FitTransformResult(model=model, transformed=Z, components=components)


def fit_ica(
    X: np.ndarray,
    n_components: int | None = None,
    random_state: int = 0,
    max_iter: int = 500,
    whiten: str = "unit-variance",
) -> FitTransformResult:
    """Independent Component Analysis (FastICA) for source separation."""
    X = np.asarray(X, dtype=float)
    model = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        whiten=whiten,
    )
    S = model.fit_transform(X)
    return FitTransformResult(
        model=model,
        transformed=S,
        components=model.components_,
    )


def ica_unmix_channels(
    multi_channel: np.ndarray,
    n_components: int | None = None,
    random_state: int = 0,
) -> FitTransformResult:
    """ICA treating rows as channels and columns as time (common EEG/EMG layout).

    Input shape ``(n_channels, n_samples)`` → ICA on transposed data so each
    sample is a time point and features are channels.
    """
    X = np.asarray(multi_channel, dtype=float)
    if X.ndim != 2:
        raise ValueError("multi_channel must be 2-D (channels × time)")
    return fit_ica(X.T, n_components=n_components, random_state=random_state)


__all__ = [
    "FitTransformResult",
    "fit_pca",
    "fit_lda",
    "fit_ica",
    "ica_unmix_channels",
]
