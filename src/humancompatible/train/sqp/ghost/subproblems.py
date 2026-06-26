"""
The two convex subproblems solved at every iterate of the Ghost-penalty SQP
method (Facchinei & Kungurtsev, 2025). Both are solved exactly, in the full
n-dimensional primal space, as written in the paper.

* ``compute_kappa``    -- the ghost-penalty relaxation κ(x), Eq. (4), via an LP.
* ``solve_subproblem`` -- the search-direction QP, Eq. (3), via ``qpsolvers``.
"""

import numpy as np
import scipy.sparse as sparse
from qpsolvers import solve_qp
from scipy.optimize import linprog


def compute_kappa(cval, cgrad, lamb, rho):
    """Ghost-penalty relaxation κ(x), Eq. (4):

        κ(x) = (1-λ) max_i {C_i(x)_+}
             + λ  min_d { max_i (C_i(x) + ∇C_i(x)^T d)_+  :  ||d||_∞ ≤ ρ }

    The inner minimization is the LP (variables t ≥ 0 and d):

        min_{t,d} t   s.t.   C_i + ∇C_i^T d ≤ t,   t ≥ 0,   -ρ ≤ d ≤ ρ .

    The relaxation inflates the right-hand side of the QP (3) just enough that
    its feasible region is always non-empty.
    """
    cval = np.asarray(cval, dtype=float)
    cgrad = np.asarray(cgrad, dtype=float)
    m, n = cgrad.shape

    term1 = (1.0 - lamb) * np.maximum(cval, 0.0).max()

    c_obj = np.zeros(n + 1)
    c_obj[0] = 1.0  # minimize t
    # -t + ∇C_i^T d ≤ -C_i
    A_ub = np.hstack([-np.ones((m, 1)), cgrad])
    b_ub = -cval
    bounds = [(0.0, None)] + [(-rho, rho)] * n

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    term2 = lamb * (res.fun if res.success else rho)

    return term1 + term2


def solve_subproblem(grad_f, cval, cgrad, kappa, beta, tau, qp_solver="osqp"):
    """Search-direction QP, Eq. (3):

        min_d  ∇F(x)^T d + (τ/2) ||d||^2
        s.t.   C(x) + ∇C(x)^T d ≤ κ(x) e            (m linear inequalities)
               ||d||_∞ ≤ β                          (box bounds)

    Cast into ``qpsolvers`` form  min ½ dᵀP d + qᵀd  s.t.  G d ≤ h,  lb ≤ d ≤ ub
    with P = τ I (diagonal), q = ∇F, G = ∇C, h = κ e − C.

    Returns the direction ``d`` as an (n,) ndarray; raises if the solver fails.
    """
    grad_f = np.asarray(grad_f, dtype=float).reshape(-1)
    cval = np.asarray(cval, dtype=float).reshape(-1)
    cgrad = np.asarray(cgrad, dtype=float)
    m, n = cgrad.shape

    P = tau * sparse.identity(n, format="csc")
    q = grad_f
    G = sparse.csc_matrix(cgrad)
    h = kappa * np.ones(m) - cval
    lb = -beta * np.ones(n)
    ub = beta * np.ones(n)

    d = solve_qp(P, q, G, h, None, None, lb, ub, solver=qp_solver)
    if d is None:
        raise RuntimeError(f"QP solver '{qp_solver}' failed to solve subproblem (3)")
    return d
