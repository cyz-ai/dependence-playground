"""Centered Kernel Alignment (Kornblith et al. 2019).

Ported from latent_compress/estimator/cka.py (numpy). CKA(X, Y) in [0, 1] =
HSIC(X, Y) / sqrt(HSIC(X, X) HSIC(Y, Y)). 0 iff independent (characteristic
kernel), 1 iff Y is an orthogonal-transform + isotropic-scale of X.

  linear_cka(X, Y) : linear kernel, closed form, O(n d^2)
  rbf_cka(X, Y)    : RBF kernel, median-heuristic bandwidth, O(n^2)
"""
import numpy as np

from ._common import as2d, pdist


def linear_cka(X, Y):
    """Linear CKA, closed form. O(n d^2)."""
    X, Y = as2d(X), as2d(Y)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = ((Y.T @ X) ** 2).sum()
    denom = np.sqrt(((X.T @ X) ** 2).sum() * ((Y.T @ Y) ** 2).sum())
    return float(np.clip(num / max(denom, 1e-12), 0.0, 1.0))


def _rbf_gram(X, sigma=None):
    """RBF Gram matrix; bandwidth via median heuristic (sigma^2 = median/2) if None."""
    sq = pdist(X) ** 2
    if sigma is None:
        off = sq[sq > 0]
        s2 = (np.median(off) / 2.0) if off.size else 1.0
        s2 = max(float(s2), 1e-12)
    else:
        s2 = sigma ** 2
    return np.exp(-sq / (2.0 * s2))


def _center_gram(K):
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def rbf_cka(X, Y, sigma_x=None, sigma_y=None):
    """RBF-kernel CKA with median-heuristic bandwidth per side. O(n^2)."""
    X, Y = as2d(X), as2d(Y)
    Kc = _center_gram(_rbf_gram(X, sigma_x))
    Lc = _center_gram(_rbf_gram(Y, sigma_y))
    num = (Kc * Lc).sum()
    denom = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(np.clip(num / max(denom, 1e-12), 0.0, 1.0))
