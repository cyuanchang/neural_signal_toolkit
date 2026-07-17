"""Demo — Section 4 PCA / LDA / ICA on a toy multi-channel mixture."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs

from neural_signal_toolkit import decomposition

# PCA / LDA on labeled clouds
X, y = make_blobs(n_samples=400, centers=3, n_features=8, random_state=0)
pca = decomposition.fit_pca(X, n_components=2)
lda = decomposition.fit_lda(X, y, n_components=2)
print("PCA explained variance:", pca.explained_variance_ratio_)
print("LDA embedding shape:", lda.transformed.shape)

# ICA on synthetic channel mixtures
rng = np.random.default_rng(0)
t = np.linspace(0, 1, 1000)
s1 = np.sin(2 * np.pi * 5 * t)
s2 = np.sign(np.sin(2 * np.pi * 7 * t))
S = np.vstack([s1, s2])
A = np.array([[1.0, 0.7], [0.4, 1.0]])
Xmix = A @ S
ica = decomposition.ica_unmix_channels(Xmix, n_components=2)
print("ICA sources shape:", ica.transformed.shape)
