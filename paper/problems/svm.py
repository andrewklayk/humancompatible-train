"""
Hard-margin SVM on a linearly separable Iris pair — the νPI §5.1 setup.

This is the *realistic* multiplier-convergence case, and the reason it is here
rather than another random QP: at the solution only the support vectors carry a
positive multiplier, so most of ``y*`` is **exactly zero**. Reaching ``y*`` then
requires a method that can drive a dual variable to zero and leave it there,
which is precisely what the raw-value quadratic penalty prevented (it pulled
strictly feasible points back onto the margin).

Formulation, in the package's ``c(x) <= 0`` convention, over ``z = (w, b)``:

    min  0.5 ||w||^2      s.t.   1 - t_i (w'x_i + b) <= 0,   i = 1..N

so ``m = N`` and the multipliers of those constraints are the usual SVM duals
``alpha``. The reference is the primal QP solved by ``qpsolvers``, with
``P = diag(I_d, 0)`` (no penalty on the bias), ``G_i = -t_i [x_i, 1]`` and
``h_i = -1``.
"""

import numpy as np
import torch

from . import Problem


def _iris_pair(class_a=0, class_b=1, standardize=True):
    """Return ``(X, t)`` for two Iris classes with labels in ``{-1, +1}``.

    Features are standardised over the selected pair. That changes nothing about
    separability but keeps ``||w*||`` at order 1, so a single primal step size
    works for every method without per-method rescaling.
    """
    from sklearn.datasets import load_iris

    data = load_iris()
    mask = np.isin(data.target, [class_a, class_b])
    X = data.data[mask].astype(float)
    t = np.where(data.target[mask] == class_a, -1.0, 1.0)
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, t


def _reference_svm(X, t, solver="clarabel"):
    """Solve the hard-margin primal and return ``(w, b, alpha, active)``."""
    import qpsolvers
    from scipy.sparse import csc_matrix

    n, d = X.shape
    P = np.zeros((d + 1, d + 1))
    P[:d, :d] = np.eye(d)                     # no quadratic term on the bias
    q = np.zeros(d + 1)
    G = -t[:, None] * np.hstack([X, np.ones((n, 1))])
    h = -np.ones(n)

    problem = qpsolvers.Problem(
        P=csc_matrix(P), q=q, G=csc_matrix(G), h=h
    )
    solution = qpsolvers.solve_problem(problem, solver=solver)
    if not solution.found:
        raise RuntimeError(f"reference SVM solve failed ({solver})")

    z = np.asarray(solution.x, dtype=float)
    residual = G @ z - h                      # <= 0 at a feasible point
    active = residual > -1e-7

    # The solver is accurate to ~1e-9, which would become the floor of every
    # ||y_k - y*|| curve and make distinct methods tie at a number that says
    # nothing about them. So use the solver only to *identify the active set*,
    # then solve the resulting equality-constrained KKT system exactly:
    #
    #     [ P    G_A' ] [ z     ]   [ 0   ]
    #     [ G_A  0    ] [ alpha ] = [ h_A ]
    #
    # which is a 9-by-9 linear system here and gives (z*, alpha*) to machine
    # precision. Inactive multipliers are exactly zero by definition.
    n_active = int(active.sum())
    G_active = G[active]
    kkt = np.block([
        [P, G_active.T],
        [G_active, np.zeros((n_active, n_active))],
    ])
    rhs = np.concatenate([np.zeros(d + 1), h[active]])
    exact = np.linalg.solve(kkt, rhs)
    z_exact, alpha_active = exact[: d + 1], exact[d + 1:]

    if not np.allclose(z, z_exact, atol=1e-6):
        raise AssertionError(
            "svm_iris: the exact active-set solution disagrees with the QP solver "
            f"(max diff {np.abs(z - z_exact).max():.2e}) — the active set is wrong"
        )
    if alpha_active.min() < 0:
        raise AssertionError(
            f"svm_iris: exact multipliers are not all nonnegative "
            f"(min {alpha_active.min():.3e}), so the active set is wrong"
        )
    if (G @ z_exact - h).max() > 1e-12:
        raise AssertionError("svm_iris: the exact solution is infeasible")

    alpha = np.zeros(len(h))
    alpha[active] = alpha_active
    return z_exact[:d], z_exact[d], alpha, active


def svm_iris(class_a=0, class_b=1, standardize=True):
    """Hard-margin SVM on Iris setosa vs versicolor (m = 100, d = 4).

    The reference multipliers are asserted unique before they are used as a
    target: with the active constraints' gradients linearly independent, the
    QP's multipliers are determined, so a nonzero ``||y_k - y*||`` is a property
    of the method and not of a degenerate reference.
    """
    X, t = _iris_pair(class_a, class_b, standardize)
    n, d = X.shape
    w_star, b_star, alpha, active = _reference_svm(X, t)

    G = -t[:, None] * np.hstack([X, np.ones((n, 1))])
    rank = np.linalg.matrix_rank(G[active], tol=1e-8)
    if rank != int(active.sum()):
        raise AssertionError(
            f"svm_iris: reference multipliers are not unique — {int(active.sum())} "
            f"active constraints but their gradients have rank {rank}"
        )
    # Separability: the reference must be strictly feasible up to solver noise.
    margin = -(G @ np.concatenate([w_star, [b_star]]) - (-np.ones(n))).min()
    if margin < -1e-7:
        raise AssertionError(f"svm_iris: classes are not separable (margin {margin:.2e})")

    dtype = torch.get_default_dtype()
    Xt = torch.as_tensor(X, dtype=dtype)
    tt = torch.as_tensor(t, dtype=dtype)

    def make_params():
        # z0 = 0: every constraint is violated by exactly 1, which is a clean,
        # method-independent starting point (and f(z0) = 0).
        return [torch.nn.Parameter(torch.zeros(d + 1, dtype=dtype))]

    def objective(params):
        w = params[0][:d]
        return 0.5 * w @ w

    def constraints(params):
        z = params[0]
        return 1.0 - tt * (Xt @ z[:d] + z[d])

    x_star = np.concatenate([w_star, [b_star]])
    return Problem(
        name="svm_iris",
        m=n,
        make_params=make_params,
        objective=objective,
        constraints=constraints,
        y_star=alpha,
        x_star=x_star,
        f_star=float(0.5 * w_star @ w_star),
        is_convex=True,
        notes=(
            f"hard-margin SVM, {int(active.sum())} of {n} constraints active, "
            f"so {n - int(active.sum())} multipliers are exactly zero"
        ),
        # The Hessian of 0.5||w||^2 is diag(I_d, 0): curvature 1 in w and *zero*
        # in the bias, so with no quadratic term the surrogate is bilinear in
        # (b, y) — see the note on plain gradient ascent in e0/a_multipliers.py.
        grad_lipschitz=1.0,
        jac_norm_sq=float(np.linalg.norm(G, 2) ** 2),
        meta={"X": X, "t": t, "G": G, "n_support": int(active.sum()), "support": active},
    )
