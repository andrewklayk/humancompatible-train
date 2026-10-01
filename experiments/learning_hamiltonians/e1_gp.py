"""Experiment 1a -- recover H for the pendulum with a Gaussian process (Sec. II, Eqs. 6-8).

No training loop. The GP interpolant from grid values is H(x) = k(x,Y) K^-1 h, so its
gradient is linear in h; stacking the gradient observations plus one pinning row gives a
(2N+1) x M least-squares system for h = H(Y).

Solved in the substituted variable u = K^-1 h -- identical answer, but it never forms K^-1,
which for a smooth kernel on a fine grid is numerically rank-deficient.
"""

import numpy as np
from pathlib import Path
from common import H_true, pendulum_data, plot_H, report

EPS = 1.0        # kernel width
RCOND = 1e-4     # least-squares cutoff; the kernel matrix is deliberately smooth
N_DATA = 400     # gradient observations
N_GRID = 25      # per axis -> M = 625, as in the paper
BOX = 2.2        # phase-space window


def k(A, B, eps=EPS):
    """Gaussian kernel, Eq. (6). A: (n,2), B: (m,2) -> (n,m)."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / eps**2)


def dk(A, B, eps=EPS):
    """d k(a,b) / d a, shape (2, n, m)."""
    diff = A[:, None, :] - B[None, :, :]
    return np.moveaxis(-2 * diff / eps**2 * k(A, B, eps)[..., None], -1, 0)


def main():
    q, p, qd, pd = pendulum_data()
    inside = (np.abs(q) < BOX) & (np.abs(p) < BOX)  # H is only recovered where data lives
    q, p, qd, pd = [a[inside] for a in (q, p, qd, pd)]
    idx = np.random.default_rng(0).choice(len(q), N_DATA, replace=False)
    X = np.stack([q[idx], p[idx]], 1)
    # g(x) = (dH/dq, dH/dp) = (-pdot, qdot), from the data alone
    g = np.stack([-pd[idx], qd[idx]], 1)

    lin = np.linspace(-BOX, BOX, N_GRID)
    Y = np.stack(np.meshgrid(lin, lin), -1).reshape(-1, 2)

    # rows: the two partials at every x_i, then the pin H(x0) = H0
    D = dk(X, Y)                              # (2, N, M)
    x0, H0 = X[:1], H_true(*X[0])
    A = np.vstack([D[0], D[1], k(x0, Y)])
    b = np.concatenate([g[:, 0], g[:, 1], [H0]])

    u, *_ = np.linalg.lstsq(A, b, rcond=RCOND)
    h = k(Y, Y) @ u

    H_ref = H_true(Y[:, 0], Y[:, 1])
    print(f"GP on {N_DATA} gradient observations, {len(Y)}-point grid:")
    H_hat = report(h, H_ref, "whole grid    ")
    # the paper's caveat: H is only recovered where the trajectories actually went
    covered = (((Y[:, None, :] - X[None, :, :]) ** 2).sum(-1).min(1) < 0.25**2)
    report(H_hat[covered], H_ref[covered], f"covered ({covered.sum():3d} pts)")
    plot_H(Y[:, 0], Y[:, 1], H_hat, H_ref, "GP", Path(__file__).parent / "figs/e1_gp.png")


if __name__ == "__main__":
    main()
