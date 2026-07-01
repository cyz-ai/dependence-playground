"""Shared helpers for the dependence measures (numpy)."""
import numpy as np


def as2d(x):
    """Coerce a 1-D vector or [n, d] array to a float [n, d] array."""
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def pdist(X):
    """Pairwise Euclidean distance matrix [n, n] for rows of X."""
    sq = (X * X).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)
