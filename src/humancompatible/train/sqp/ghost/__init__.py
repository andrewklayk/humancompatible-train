"""
Stochastic Ghost-penalty SQP for expectation-objective, expectation-inequality
constrained nonconvex optimization.

Implementation of Facchinei & Kungurtsev (2025), "Stochastic Approximation for
Expectation Objective and Expectation Inequality-Constrained Nonconvex
Optimization" (arXiv:2307.02943): the SCP subproblems (Eqs. 3-4), the unbiased
randomized-MLMC step estimator (Eq. 23) and the diminishing-step SA driver
(Section 2).
"""

from .ghost_sqp import GhostConfig, GhostSQP
from .mlmc import mlmc_direction
from .oracle import evaluate_oracle, n_params
from .sampler import Sampler, TensorSampler
from .subproblems import compute_kappa, solve_subproblem

__all__ = [
    "GhostConfig",
    "GhostSQP",
    "mlmc_direction",
    "evaluate_oracle",
    "n_params",
    "Sampler",
    "TensorSampler",
    "compute_kappa",
    "solve_subproblem",
]
