"""
Correlation is not dependence: concise 1x7 figure.

Shows a linear Gaussian (rho=0.8) followed by 6 nonlinear dependencies
where Pearson correlation ≈ 0 but MI > 0.

Reference: https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


def estimate_mi(x, y, n_neighbors=5):
    """Estimate MI(X; Y) using KSG via sklearn."""
    return mutual_info_regression(
        x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=0
    )[0]


def myplot(ax, xy, xlim=(-4, 4), ylim=(-4, 4), eps=1e-15):
    x, y = xy[:, 0], xy[:, 1]
    rho = np.corrcoef(x, y)[0, 1] if np.std(y) > eps else np.nan
    mi = estimate_mi(x, y) if np.std(y) > eps else 0.0
    rho_str = f"ρ={rho:.2f}" if not np.isnan(rho) else ""
    mi_str = f"MI={mi:.2f}"
    ax.scatter(x, y, c="darkblue", s=0.5, edgecolors="none")
    ax.set_title(f"{rho_str}\n{mi_str}", fontsize=9, linespacing=1.2)
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


def main(save_path=None):
    fig, axes = plt.subplots(1, 7, figsize=(12, 2))
    fig.subplots_adjust(wspace=0.25)

    np.random.seed(42)
    n = 800

    # 1. Linear Gaussian (rho=0.8)
    cov = np.array([[1, 0.8], [0.8, 1]])
    xy = np.random.multivariate_normal([0, 0], cov, n)
    myplot(axes[0], xy)

    # 2. Double parabola (W-shape)
    x = np.random.uniform(-1, 1, n)
    y = 4 * (x**2 - 0.5)**2 + np.random.uniform(-1, 1, n) / 3
    myplot(axes[1], np.column_stack([x, y]), xlim=(-1, 1), ylim=(-1/3, 1+1/3))

    # 3. Double-rotated uniform (diamond)
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    xy = rotation(-np.pi/4, np.column_stack([x, y]))
    myplot(axes[2], xy, xlim=(-np.sqrt(2), np.sqrt(2)), ylim=(-np.sqrt(2), np.sqrt(2)))

    # 4. Parabola
    x = np.random.uniform(-1, 1, n)
    y = 2 * x**2 + np.random.uniform(-1, 1, n)
    myplot(axes[3], np.column_stack([x, y]), xlim=(-1, 1), ylim=(-1, 3))

    # 5. X-shape
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

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main(save_path="correlation_vs_mi_concise.png")
