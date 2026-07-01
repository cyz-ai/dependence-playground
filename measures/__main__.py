"""Selftest for the dependence measures: `python -m measures`."""
import numpy as np

from . import distance_correlation, linear_cka, rbf_cka, cosine

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    x = rng.standard_normal((n, 3))
    xi, yi = rng.standard_normal((n, 3)), rng.standard_normal((n, 3))
    z = rng.standard_normal((n, 3)); w = z + 0.3 * rng.standard_normal((n, 3))
    print(f"{'case':12s} {'dCor':>6s} {'linCKA':>7s} {'rbfCKA':>7s} {'cosine':>7s}")
    for name, (a, b) in {"identical": (x, x), "independent": (xi, yi), "dependent": (z, w)}.items():
        print(f"{name:12s} {distance_correlation(a,b):6.3f} {linear_cka(a,b):7.3f} "
              f"{rbf_cka(a,b):7.3f} {cosine(a,b):7.3f}")
