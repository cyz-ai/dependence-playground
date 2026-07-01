"""Distance correlation (Székely & Rizzo).

Ported from latent_compress/estimator/dcorr.py (numpy). Bias-corrected /
unbiased (U-statistic) estimator: dCor(X, Y) in [0, 1], = 0 iff X and Y are
independent. Detects nonlinear dependence that Pearson rho misses.
"""
import numpy as np

from ._common import as2d, pdist


def distance_correlation(x, y):
    """Unbiased (U-statistic) distance correlation, Székely & Rizzo (2014).

    U-centered distance matrices ((m-2), (m-1)(m-2) divisors; m(m-3) norm) with the
    DIAGONAL zeroed -- the inner product runs over i != j. Omitting the zeroing leaks
    the diagonal in and inflates the estimate (independent data reads ~0.1-0.25 instead
    of ~0). The unbiased dCov^2 can be slightly negative under independence, so it is
    clamped to 0 before the square root.
    """
    X, Y = as2d(x), as2d(y)
    m = X.shape[0]
    A, B = pdist(X), pdist(Y)
    a = A - A.sum(0, keepdims=True) / (m - 2) - A.sum(1, keepdims=True) / (m - 2) + A.sum() / ((m - 1) * (m - 2))
    b = B - B.sum(0, keepdims=True) / (m - 2) - B.sum(1, keepdims=True) / (m - 2) + B.sum() / ((m - 1) * (m - 2))
    np.fill_diagonal(a, 0.0)                                  # U-centering: diagonal must be 0
    np.fill_diagonal(b, 0.0)
    AB = (a * b).sum() / (m * (m - 3))
    AA = (a * a).sum() / (m * (m - 3))
    BB = (b * b).sum() / (m * (m - 3))
    denom = np.sqrt(max(AA, 0.0) * max(BB, 0.0))
    if denom <= 1e-12:
        return 0.0
    return float(np.sqrt(max(AB, 0.0)) / np.sqrt(denom))     # clamp: unbiased dCov^2 may be < 0
