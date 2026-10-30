"""
The two convex subproblems solved at every iterate of the Ghost-penalty SQP
method (Facchinei & Kungurtsev, 2025). Both are solved exactly, in the full
n-dimensional primal space, as written in the paper.

* ``compute_kappa``    -- the ghost-penalty relaxation κ(x), Eq. (4), via an LP.
* ``solve_subproblem`` -- the search-direction QP, Eq. (3), via ``qpsolvers``.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sparse
from qpsolvers import solve_qp
# TODO: unused import
import torch
from scipy.optimize import linprog


def compute_kappa(
        cval: np.ndarray,
        cgrad: np.ndarray,
        lamb: float,
        rho: float
    ) -> float:
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

    # TODO: skip LP if cval.max() <= 0 (kappa = 0 exactly)
    term1 = (1.0 - lamb) * np.maximum(cval, 0.0).max()

    c_obj = np.zeros(n + 1)
    c_obj[0] = 1.0  # minimize t
    # -t + ∇C_i^T d ≤ -C_i
    A_ub = np.hstack([-np.ones((m, 1)), cgrad])
    b_ub = -cval
    # TODO: pass (n+1, 2) array, not list of n tuples
    bounds = [(0.0, None)] + [(-rho, rho)] * n

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    term2 = lamb * (res.fun if res.success else rho)

    return term1 + term2


def solve_subproblem(
        grad_f: np.ndarray,
        cval: np.ndarray,
        cgrad: np.ndarray,
        kappa: float,
        beta: float,
        tau: float,
        qp_solver: str = "osqp",
        solve_dual: bool = False,
        solve_relaxed: bool = False
    )-> Optional[np.ndarray]:
    """Search-direction QP, Eq. (3):

        min_d  ∇F(x)^T d + (τ/2) ||d||^2
        s.t.   C(x) + ∇C(x)^T d ≤ κ(x) e            (m linear inequalities)
               ||d||_∞ ≤ β                          (box bounds)

    Cast into ``qpsolvers`` form  min ½ dᵀQ d + cᵀd  s.t.  A d ≤ b,  lb ≤ d ≤ ub
    with Q = τ I (diagonal), c = ∇F, A = ∇C, b = κ e − C.

    Returns the direction ``d`` as an (n,) ndarray; raises if the solver fails.
    """
    
    # prepare the needed results
    grad_f = np.asarray(grad_f, dtype=float).reshape(-1)
    cval = np.asarray(cval, dtype=float).reshape(-1)
    cgrad = np.asarray(cgrad, dtype=float)
    m, n = cgrad.shape
    d = None

    # TODO: build Q,c,A,b once; 4 branches are copy-paste (b recomputed, Q unused in dual)
    if solve_dual:  # solve the dual problem

        if solve_relaxed: # solve the dual problem with no beta constraints - relaxation
            # create the primal problem
            Q = tau * sparse.identity(n, format="csc")
            c = grad_f
            # TODO: keep A dense here; Q_dual is m x m
            A = sparse.csc_matrix(cgrad)
            b = kappa * np.ones(m) - cval

            # create the dual problem
            Q_dual = A @ A.T / tau
            c_dual = A @ c / tau + b
            A_dual = -sparse.identity(m, format="csc")
            b_dual = np.zeros(m)

            # solve the dual problem
            lambda_optimal = solve_qp(Q_dual, c_dual, A_dual, b_dual, None, None, None, None, solver=qp_solver)

            # recover the primal solution from the dual solution
            d = -1/tau * (A.T @ lambda_optimal + c)
        
        # TODO: dual has m+2n vars > n primal: never cheaper, drop or fall back to primal
        else:   # solve the dual problem with beta constraints
            # create the primal problem
            Q = tau * sparse.identity(n, format="csc")
            c = grad_f
            A = sparse.csc_matrix(cgrad)
            b = kappa * np.ones(m) - cval

            # append the box constraints (beta)
            A = sparse.vstack([A, sparse.identity(n), -sparse.identity(n)], format="csc")
            b = kappa * np.ones(m) - cval
            b = np.concatenate([b, beta * np.ones(n), beta * np.ones(n)])

            # create the dual problem
            Q_dual = A @ A.T / tau
            c_dual = A @ c / tau + b
            A_dual = -sparse.identity(m + 2*n, format="csc")
            b_dual = np.zeros(m + 2*n)

            # solve the dual problem
            lambda_optimal = solve_qp(Q_dual, c_dual, A_dual, b_dual, None, None, None, None, solver=qp_solver)

            # recover the primal solution from the dual solution
            d = -1/tau * (A.T @ lambda_optimal + c)


    else: # solve the primal problem

        if solve_relaxed:  # if relaxation is applied - beta is ignored in the subproblem - faster solution
            Q = tau * sparse.identity(n, format="csc")
            c = grad_f
            A = sparse.csc_matrix(cgrad)
            b = kappa * np.ones(m) - cval
            d = solve_qp(Q, c, A, b, None, None, None, None, solver=qp_solver)

        else:  # beta is not ignored in the subproblem - slower solution, but equal to the proposed stochastic ghost method
            Q = tau * sparse.identity(n, format="csc")
            c = grad_f
            A = sparse.csc_matrix(cgrad)
            b = kappa * np.ones(m) - cval
            lb = -beta * np.ones(n)
            ub = beta * np.ones(n)
            d = solve_qp(Q, c, A, b, None, None, lb, ub, solver=qp_solver)

    return d
