"""Dependence measures ported from the latent_compress project (numpy).

One file per measure. Each takes X, Y (1-D vectors or [n, d] arrays, paired rows
x_i <-> y_i) and returns a Python float. Use alongside Pearson rho and MI to show
"correlation != dependence": unlike rho, distance correlation and CKA detect
nonlinear dependence (rho~0 but dCor/CKA > 0 for parabola / circle / X-shape).

    from measures import distance_correlation, linear_cka, rbf_cka, cosine
"""
from .dcorr import distance_correlation
from .cka import linear_cka, rbf_cka
from .cosine import cosine

__all__ = ["distance_correlation", "linear_cka", "rbf_cka", "cosine"]
