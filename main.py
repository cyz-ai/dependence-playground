"""
Correlation is not dependence: a 3x7 grid of scatterplots.

Recreates the classic visualization by Denis Boigelot / Imagecreator
(Wikipedia: Correlation and dependence). Each panel shows (x, y) pairs
with both Pearson correlation (rho) and mutual information (MI) in the title.
Row 3 demonstrates that many structured dependencies have rho ≈ 0 but MI > 0.

Reference: https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

from measures import distance_correlation, rbf_cka


def estimate_mi(x, y, n_neighbors=5):
    """Estimate MI(X; Y) using KSG (k-nearest neighbors) via sklearn."""
    return mutual_info_regression(
        x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=0
    )[0]


def myplot(ax, xy, xlim=(-4, 4), ylim=(-4, 4), eps=1e-15):
    x, y = xy[:, 0], xy[:, 1]
    valid = np.std(y) > eps
    rho = np.corrcoef(x, y)[0, 1] if valid else np.nan
    mi = estimate_mi(x, y) if valid else 0.0
    # dependence measures ported from latent_compress (rho~0 but these > 0 for
    # nonlinear dependence -> "correlation is not dependence")
    dcor = distance_correlation(x, y) if valid else 0.0
    cka = rbf_cka(x, y) if valid else 0.0
    # Title: rho + MI on top, distance-correlation + RBF-CKA below
    rho_str = f"ρ={rho:.2f}" if not np.isnan(rho) else "ρ=—"
    ax.scatter(x, y, c="darkblue", s=0.5, edgecolors="none")
    ax.set_title(f"{rho_str}  MI={mi:.2f}\ndCor={dcor:.2f}  CKA={cka:.2f}",
                 fontsize=7, linespacing=1.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def rotation(t, X):
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t),  np.cos(t)]])
    return X @ R.T


def mv_normal(axes, n=800):
    cors = [1.0, 0.8, 0.4, 0.0, -0.4, -0.8, -1.0]
    for ax, rho in zip(axes, cors):
        cov = np.array([[1, rho], [rho, 1]])
        xy = np.random.multivariate_normal([0, 0], cov, n)
        myplot(ax, xy)


def rot_normal(axes, n=200):
    cov = np.array([[1, 1], [1, 1]]) + 1e-6 * np.eye(2)
    xy = np.random.multivariate_normal([0, 0], cov, n)
    angles = [0, np.pi/12, np.pi/6, np.pi/4,
              np.pi/2 - np.pi/6, np.pi/2 - np.pi/12, np.pi/2]
    for ax, t in zip(axes, angles):
        myplot(ax, rotation(t, xy))


def others(axes, n=800):
    # 1. Double parabola
    x = np.random.uniform(-1, 1, n)
    y = 4 * (x**2 - 0.5)**2 + np.random.uniform(-1, 1, n) / 3
    myplot(axes[0], np.column_stack([x, y]), xlim=(-1, 1), ylim=(-1/3, 1+1/3))

    # 2. Rotated uniform
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    xy = rotation(-np.pi/8, np.column_stack([x, y]))
    lim = np.sqrt(2 + np.sqrt(2)) / np.sqrt(2)
    myplot(axes[1], xy, xlim=(-lim, lim), ylim=(-lim, lim))

    # 3. Double-rotated uniform
    xy = rotation(-np.pi/8, xy)
    myplot(axes[2], xy, xlim=(-np.sqrt(2), np.sqrt(2)), ylim=(-np.sqrt(2), np.sqrt(2)))

    # 4. Parabola
    x = np.random.uniform(-1, 1, n)
    y = 2 * x**2 + np.random.uniform(-1, 1, n)
    myplot(axes[3], np.column_stack([x, y]), xlim=(-1, 1), ylim=(-1, 3))

    # 5. X-shape (diamond)
    x = np.random.uniform(-1, 1, n)
    y = (x**2 + np.random.uniform(0, 0.5, n)) * np.random.choice([-1, 1], n)
    myplot(axes[4], np.column_stack([x, y]), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

    # 6. Circle
    x_orig = np.random.uniform(-1, 1, n)
    y = np.cos(x_orig * np.pi) + np.random.normal(0, 1/8, n)
    x = np.sin(x_orig * np.pi) + np.random.normal(0, 1/8, n)
    myplot(axes[5], np.column_stack([x, y]), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))

    # 7. Four Gaussian clusters
    xy1 = np.random.multivariate_normal([3, 3], np.eye(2), n // 4)
    xy2 = np.random.multivariate_normal([-3, 3], np.eye(2), n // 4)
    xy3 = np.random.multivariate_normal([-3, -3], np.eye(2), n // 4)
    xy4 = np.random.multivariate_normal([3, -3], np.eye(2), n // 4)
    xy = np.vstack([xy1, xy2, xy3, xy4])
    myplot(axes[6], xy, xlim=(-7, 7), ylim=(-7, 7))


def main(save_path=None):
    fig, axes = plt.subplots(3, 7, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.4, hspace=0.7)

    np.random.seed(42)
    mv_normal(axes[0])
    rot_normal(axes[1])
    others(axes[2])

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main(save_path="correlation_vs_mi.png")
