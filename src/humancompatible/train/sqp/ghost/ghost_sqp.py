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

    alpha: float = 0.3      # geometric level parameter p_α ∈ (0, 1)
    base_batch: int = 1     # level-0 sample count m0; level ℓ uses m0·2^ℓ samples
    tau: float = 1.0        # QP regularization τ > 0
    beta: float = 10.0      # trust-region bound ||d||_∞ ≤ β
    rho: float = 0.8        # ghost-penalty radius ρ ∈ (0, β)
    lambda_: float = 0.5    # ghost-penalty weight λ ∈ (0, 1)
    gamma0: float = 0.9     # step-size scale; γ^ν = γ0 / ν^p  (γ0 < 1 ⇒ γ^1 < 1)
    gamma_power: float = 1.0  # decay exponent p; any p ∈ (0.5, 1] gives Σγ=∞, Σγ²<∞
    qp_solver: str = "osqp"


class GhostSQP:
    """Driver for the stochastic Ghost-penalty SQP method.

    Parameters
    ----------
    net : torch.nn.Module
        Model whose parameters x are optimized in place.
    objective_fn : callable(net, batch) -> scalar tensor
        Mean objective loss over the batch.
    constraint_fns : sequence of callable(net, batch) -> scalar tensor
        Each returns the mean of a constraint ``c_i`` (interpreted ``≤ 0``).
    sampler : Sampler
        Source of fresh i.i.d. minibatches (see :mod:`.sampler`).
    cfg : GhostConfig, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(self, net, objective_fn, constraint_fns, sampler, cfg=None, rng=None):
        self.net = net
        self.objective_fn = objective_fn
        self.constraint_fns = list(constraint_fns)
        self.sampler = sampler
        self.cfg = cfg or GhostConfig()
        self.rng = rng if rng is not None else np.random.defgradlt_rng()
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
            self.sampler, self.cfg, self.rng,
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
