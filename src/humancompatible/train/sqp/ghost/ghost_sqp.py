"""
Stochastic Ghost-penalty SQP driver (Facchinei & Kungurtsev, 2025).

Solves the expectation-constrained nonconvex problem

    min_x  F(x) = E[f(x, ξ)]   s.t.   C_i(x) = E[c_i(x, ζ)] ≤ 0,  i = 1..m

by the diminishing-step stochastic-approximation scheme of Section 2: at each
major iteration ν it forms an unbiased MLMC estimate d̃(x^ν) of the SCP search
direction (Eq. 23) and takes the step

    x^{ν+1} = x^ν + γ^ν d̃(x^ν),       γ^ν = γ0 / ν

with γ^ν satisfying Σ γ^ν = ∞ and Σ (γ^ν)^2 < ∞ (and γ^1 < 1 for γ0 < 1).

No penalty parameter is tuned and no line search is performed; under MFCQ and
the paper's assumptions every limit point is a KKT point (Theorem 2.1).
"""

from dataclasses import dataclass

import numpy as np
import torch

from .mlmc import mlmc_direction


@dataclass
class GhostConfig:
    """Hyperparameters for :class:`GhostSQP`."""

    alpha: float | None = None      # geometric level parameter p_α ∈ (0, 1)
    base_batch: int = 1     # level-0 sample count m0; level ℓ uses m0·2^ℓ samples
    tau: float = 1.0        # QP regularization τ > 0
    beta: float | None = None      # trust-region bound ||d||_∞ ≤ β
    rho: float = 0.8        # ghost-penalty radius ρ ∈ (0, β)
    lambda_: float = 0.5    # ghost-penalty weight λ ∈ (0, 1)
    gamma0: float = 0.9     # step-size scale; γ^ν = γ0 / ν^p  (γ0 < 1 ⇒ γ^1 < 1)
    gamma_power: float = 1.0  # decay exponent p; any p ∈ (0.5, 1] gives Σγ=∞, Σγ²<∞
    qp_solver: str = "osqp"
    solve_dual: bool = True # whether to solve the primal or dual problem
    solve_relaxed: bool = True # whether to ignore the beta constraint in the subproblem (faster solution, but not equal to the proposed stochastic ghost method)
    monte_carlo_sampling: bool = False # whether to use monte carlo sampling or not (if false, uses base_batch samples) 

    def __post_init__(self):
        if self.alpha is not None and self.monte_carlo_sampling is False:
            raise ValueError("alpha should be None if monte_carlo_sampling is False, the estimate is based only on base_batch samples")
        if self.alpha is not None and not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1) if monte_carlo_sampling is True, got {self.alpha}")
        if self.alpha is None and self.monte_carlo_sampling is True:
            self.alpha = 0.9
            print(f"Warning: alpha is not set, using default value of {self.alpha}")


        if self.beta is not None and self.solve_relaxed is True:
            raise ValueError("beta should be None if solve_relaxed is True, the beta constraint is ignored in the subproblem")
        if self.beta is None and self.solve_relaxed is False: 
            self.beta = 10.0
            print(f"Warning: beta is not set, using default value of {self.beta}")
        

        if self.beta is not None and self.beta < self.rho:
            raise ValueError(f"beta must be greater than or equal to rho, got beta={self.beta}, rho={self.rho}")

        if self.tau <= 0.0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if self.beta is not None and self.beta <= 0.0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.rho <= 0.0:
            raise ValueError(f"rho must be positive, got {self.rho}")
        if self.lambda_ <= 0.0 or self.lambda_ >= 1.0:
            raise ValueError(f"lambda_ must be in (0, 1), got {self.lambda_}")

    

class GhostSQP:
    """Driver for the stochastic Ghost-penalty SQP method.

    Parameters
    ----------
    net : torch.nn.Module
        Model whose parameters x are optimized in place.
    objective_fn : callable(net, batch) -> scalar tensor
        Mean objective loss over the batch.
    constraint_fns : sequence of callable(net, batch) -> scalar tensor, or
        a single callable(net, batch) -> (m,) tensor
        Each returns the mean of a constraint ``c_i`` (interpreted ``≤ 0``).
        Pass the vector form for large ``m``: see :func:`.oracle.evaluate_oracle`.
    sampler : Sampler
        Source of fresh i.i.d. minibatches (see :mod:`.sampler`).
    cfg : GhostConfig, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, net, objective_fn, constraint_fns, sampler, cfg=None, rng=None):
        self.net = net
        self.objective_fn = objective_fn
        self.constraint_fns = constraint_fns if callable(constraint_fns) else list(constraint_fns)
        self.sampler = sampler
        self.cfg = cfg or GhostConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._nu = 0



    def gamma(self, nu):
        """Diminishing step size γ^ν = γ0 / ν^p."""
        return self.cfg.gamma0 / nu ** self.cfg.gamma_power

    @torch.no_grad()
    def _apply_direction(self, d, gamma):
        """In-place update x ← x + γ d, unflattening d across parameters."""
        i = 0
        for p in self.net.parameters():
            k = p.numel()
            chunk = torch.as_tensor(d[i:i + k], dtype=p.dtype, device=p.device).view_as(p)
            p.add_(chunk, alpha=gamma)
            i += k

    def step(self):
        """Perform one major SA iteration; returns a diagnostics dict."""
        self._nu += 1
        gamma = self.gamma(self._nu)
        d, info = mlmc_direction(
            self.net, self.objective_fn, self.constraint_fns,
            self.sampler, self.cfg, self.rng, solve_dual = self.cfg.solve_dual, 
            solve_relaxed=self.cfg.solve_relaxed, monte_carlo_sampling=self.cfg.monte_carlo_sampling
        )
        self._apply_direction(d, gamma)
        info.update(iter=self._nu, gamma=gamma, step_norm=float(np.linalg.norm(d)))
        return info

    def train(self, n_iters, callback=None):
        """Run ``n_iters`` SA iterations; returns the list of per-step infos."""
        history = []
        for _ in range(n_iters):
            info = self.step()
            history.append(info)
            if callback is not None:
                callback(info)
        return history
