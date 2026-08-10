"""
Quadratic programs with known KKT points, for validating multiplier convergence.

Two convex instances are *constructed* so that ``(x*, y*)`` are known in closed
form rather than read off a solver, and then cross-checked against a QP solver;
that way a disagreement points at the construction or the solver rather than
leaving both unverified. The nonconvex instance has no reference multipliers by
construction, so only the KKT residual at the returned iterate is meaningful.

Convention throughout: constraints are ``A x - b <= 0``, so the Lagrangian is
``f(x) + y'(Ax - b)`` and stationarity reads ``Qx + q + A'y = 0``.
"""

import numpy as np
import torch

from . import Problem


def _reference_solution(Q, q, A, b, solver="clarabel"):
    """Solve the QP and return ``(x, y)`` with ``y`` the multipliers of ``Ax <= b``."""
    import qpsolvers
    from scipy.sparse import csc_matrix

    problem = qpsolvers.Problem(
        P=csc_matrix(np.asarray(Q, dtype=float)),
        q=np.asarray(q, dtype=float),
        G=csc_matrix(np.asarray(A, dtype=float)),
        h=np.asarray(b, dtype=float),
    )
    solution = qpsolvers.solve_problem(problem, solver=solver)
    if not solution.found:
        raise RuntimeError(f"reference QP solve failed ({solver})")
    return np.asarray(solution.x, dtype=float), np.asarray(solution.z, dtype=float)


def _torch_qp(name, Q, q, A, b, *, y_star, x_star, is_convex, x0, notes):
    Qt = torch.as_tensor(Q, dtype=torch.get_default_dtype())
    qt = torch.as_tensor(q, dtype=torch.get_default_dtype())
    At = torch.as_tensor(A, dtype=torch.get_default_dtype())
    bt = torch.as_tensor(b, dtype=torch.get_default_dtype())
    x0t = torch.as_tensor(x0, dtype=torch.get_default_dtype())

    def make_params():
        return [torch.nn.Parameter(x0t.clone())]

    def objective(params):
        x = params[0]
        return 0.5 * x @ (Qt @ x) + qt @ x

    def constraints(params):
        return At @ params[0] - bt

    f_star = None
    if x_star is not None:
        xs = torch.as_tensor(x_star, dtype=torch.get_default_dtype())
        f_star = float(0.5 * xs @ (Qt @ xs) + qt @ xs)

    return Problem(
        name=name,
        m=A.shape[0],
        make_params=make_params,
        objective=objective,
        constraints=constraints,
        y_star=y_star,
        x_star=x_star,
        f_star=f_star,
        is_convex=is_convex,
        notes=notes,
        # |eig| rather than the largest eigenvalue, so the nonconvex instance
        # gets a step size bounded by its true curvature in both directions.
        grad_lipschitz=float(np.abs(np.linalg.eigvalsh(Q)).max()),
        jac_norm_sq=float(np.linalg.norm(A, 2) ** 2),
        meta={"Q": Q, "q": q, "A": A, "b": b},
    )


def qp_active(n=10, m=5, seed=0, y_scale=1.0):
    """Strictly convex QP whose every constraint is active at the solution.

    Constructed backwards from a chosen ``(x*, y*)``: pick ``A``, set ``b = A x*``
    so all constraints hold with equality, then set ``q = -Q x* - A' y*`` so that
    stationarity holds exactly. With ``Q > 0`` the minimiser is unique, and with
    ``A`` of full row rank so are the multipliers.

    This is the clean case for multiplier convergence: every ``y*_i > 0``, so no
    method is asked to represent an exact zero.
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    Q = B @ B.T + n * np.eye(n)          # well-conditioned, positive definite
    A = rng.standard_normal((m, n))
    x_star = rng.standard_normal(n)
    y_star = y_scale * (1.0 + rng.random(m))   # strictly positive
    b = A @ x_star
    q = -Q @ x_star - A.T @ y_star

    # Cross-check the construction against an independent solver.
    x_solver, y_solver = _reference_solution(Q, q, A, b)
    if not np.allclose(x_star, x_solver, atol=1e-6):
        raise AssertionError(
            f"qp_active: constructed x* disagrees with the solver "
            f"(max diff {np.abs(x_star - x_solver).max():.2e})"
        )
    if not np.allclose(y_star, y_solver, atol=1e-6):
        raise AssertionError(
            f"qp_active: constructed y* disagrees with the solver "
            f"(max diff {np.abs(y_star - y_solver).max():.2e})"
        )

    return _torch_qp(
        "qp_active", Q, q, A, b,
        y_star=y_star, x_star=x_star, is_convex=True,
        x0=np.zeros(n),
        notes="convex QP, all constraints active, all y* > 0",
    )


def qp_inactive(n=10, m=6, n_active=2, seed=1):
    """Strictly convex QP where only some constraints are active.

    The inactive constraints have ``y*_i = 0`` exactly, which is what the old
    raw-value quadratic penalty could not represent: it was minimised by driving
    every ``c_i`` to 0, dragging strictly feasible constraints onto the boundary.
    A method that reaches ``y*`` here is one whose augmented term is right.
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    Q = B @ B.T + n * np.eye(n)
    A = rng.standard_normal((m, n))
    x_star = rng.standard_normal(n)

    y_star = np.zeros(m)
    y_star[:n_active] = 1.0 + rng.random(n_active)

    b = A @ x_star
    b[n_active:] += 0.5 + rng.random(m - n_active)   # slack: strictly feasible
    q = -Q @ x_star - A.T @ y_star

    x_solver, y_solver = _reference_solution(Q, q, A, b)
    if not np.allclose(x_star, x_solver, atol=1e-6):
        raise AssertionError(
            f"qp_inactive: constructed x* disagrees with the solver "
            f"(max diff {np.abs(x_star - x_solver).max():.2e})"
        )
    if not np.allclose(y_star, y_solver, atol=1e-6):
        raise AssertionError(
            f"qp_inactive: constructed y* disagrees with the solver "
            f"(max diff {np.abs(y_star - y_solver).max():.2e})"
        )

    return _torch_qp(
        "qp_inactive", Q, q, A, b,
        y_star=y_star, x_star=x_star, is_convex=True,
        x0=np.zeros(n),
        notes=f"convex QP, {n_active}/{m} constraints active, the rest have y*=0",
    )


def qp_nonconvex(n=5, seed=2):
    """Indefinite QP over a box, so it is bounded but has no reference multipliers.

    Constraints are ``-1 <= x_i <= 1`` written as ``2n`` inequalities. There is no
    a-priori ``y*``, so E0a reports the KKT residual at the returned iterate
    instead of a distance to reference multipliers.
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    Q = (B + B.T) / 2.0                       # symmetric, indefinite
    eigenvalues = np.linalg.eigvalsh(Q)
    if eigenvalues.min() >= 0:
        raise AssertionError("qp_nonconvex: Q came out positive semidefinite")
    q = rng.standard_normal(n)
    A = np.vstack([np.eye(n), -np.eye(n)])    # x <= 1, -x <= 1
    b = np.ones(2 * n)

    return _torch_qp(
        "qp_nonconvex", Q, q, A, b,
        y_star=None, x_star=None, is_convex=False,
        x0=np.zeros(n),
        notes=f"indefinite QP over a box, lambda_min(Q) = {eigenvalues.min():.3f}",
    )
