"""Mean cosine similarity.

Ported from latent_compress/estimator/cosine.py (numpy). Mean over rows of the
per-sample cosine similarity between paired vectors x_i and y_i.
"""
import numpy as np

from ._common import as2d


def cosine(x, y):
    """Mean per-row cosine similarity between paired rows of X and Y."""
    X, Y = as2d(x), as2d(y)
    xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    yn = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-6
    return float(((X / xn) * (Y / yn)).sum(1).mean())
