"""
Stochastic ghost-penalty SQP (Facchinei & Kungurtsev, 2025).

Solves the expectation-constrained nonconvex problem

    min_x  F(x) = E[f(x, ξ)]   s.t.   C_i(x) = E[c_i(x, ζ)] ≤ 0,  i = 1..m

with the diminishing-step scheme of Section 2: at major iteration ν it estimates
the SCP search direction d̃(x^ν)  and takes the step

    x^{ν+1} = x^ν + γ^ν d̃(x^ν),       γ^ν = γ0 / ν^p.

For p ∈ (0.5, 1], Σ γ^ν = ∞ and Σ (γ^ν)^2 < ∞; γ0 < 1 gives γ^1 < 1. No penalty
parameter is tuned and no line search is performed.

The method of the paper corresponds to
``GhostConfig(monte_carlo_sampling=True, solve_relaxed=False)``: the direction is
an unbiased MLMC estimate, the subproblem keeps the trust region, and under MFCQ
and the paper's assumptions every limit point is a KKT point (Theorem 2.1). The
defaults are a faster variant that estimates the direction from a single batch
and drops the trust region; Theorem 2.1 does not cover it.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .mlmc import direction


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
    r"""Driver for the stochastic ghost-penalty SQP method.

    **In your own training loop (recommended).** Compute the objective and the
    constraints on your batch and pass them to :meth:`step`::

        solver = GhostSQP(model, objective_fn=None, constraint_fns=None,
                          sampler=None, cfg=cfg)
        for batch in loader:
            loss = objective(model, batch)
            constraints = constraint_values(model, batch)
            solver.step(loss, constraints)

    **As a self-contained loop.** Pass ``objective_fn``, ``constraint_fns`` and
    ``sampler``; :meth:`step` then draws its own batches and :meth:`train` runs
    the iterations. MLMC (``cfg.monte_carlo_sampling=True``) requires this mode.

    :param net: Model whose trainable parameters are updated in place.
    :type net: torch.nn.Module
    :param objective_fn: ``callable(net, batch) -> scalar tensor`` returning the
        mean objective over the batch, or ``None`` to pass the loss to
        :meth:`step`.
    :type objective_fn: Callable | None; defaults to None
    :param constraint_fns: One ``callable(net, batch) -> scalar tensor`` per
        constraint :math:`c_i \le 0`, or a single
        ``callable(net, batch) -> (m,) tensor`` (preferable for large :math:`m`;
        see :func:`.oracle.evaluate_oracle`), or ``None`` to pass the
        constraints to :meth:`step`.
    :type constraint_fns: Callable | Sequence[Callable] | None; defaults to None
    :param sampler: Source of fresh i.i.d. mini-batches (see :mod:`.sampler`);
        used only when ``objective_fn`` and ``constraint_fns`` are given.
    :type sampler: Sampler | None; defaults to None
    :param cfg: Hyperparameters; defaults to ``GhostConfig()``.
    :type cfg: GhostConfig | None
    :param rng: Random generator used by the direction estimator; defaults to
        ``numpy.random.default_rng()``.
    :type rng: numpy.random.Generator | None
    """

    def __init__(
            self,
            net: torch.nn.Module,
            objective_fn: Callable[..., torch.Tensor] | None = None,
            constraint_fns: Callable[..., torch.Tensor] | Sequence[Callable[..., torch.Tensor]] | None = None,
            sampler = None,
            cfg: GhostConfig | None = None,
            rng=None
        ):
        self.net = net
        # TODO: set self.cfg first and validate it; list() conversion unneeded
        if (objective_fn is None or constraint_fns is None) and cfg is not None and cfg.monte_carlo_sampling:
            raise ValueError("MLMC direction estimation requires both objective and constraints to be passed as functions.")
        # allow None for both objective and constraint functions to allow user to supply calculated tensors (when not using mlmc)
        self.objective_fn = objective_fn
        self.constraint_fns = constraint_fns if (callable(constraint_fns) or constraint_fns is None) else list(constraint_fns)
        self.sampler = sampler
        self.cfg = cfg or GhostConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._nu = 0



    def gamma(self, nu):
        """Diminishing step size γ^ν = γ0 / ν^p."""
        return self.cfg.gamma0 / nu ** self.cfg.gamma_power

    @torch.no_grad()
    # TODO: use parameters_to_vector/vector_to_parameters with one tensor conversion
    def _apply_direction(self, d, gamma):
        """In-place update x ← x + γ d, unflattening d across parameters."""
        i = 0
        for p in self.net.parameters():
            if not p.requires_grad:
                continue  # d covers trainable params only (see oracle.n_params)
            k = p.numel()
            chunk = torch.as_tensor(d[i:i + k], dtype=p.dtype, device=p.device).view_as(p)
            p.add_(chunk, alpha=gamma)
            i += k

    def step(self, loss=None, constraints=None):
        """Perform one major SA iteration; returns a diagnostics dict."""
        self._nu += 1
        gamma = self.gamma(self._nu)
        d, info = direction(
            net=self.net,
            objective=loss,
            constraints=constraints,
            objective_fn=self.objective_fn,
            constraint_fns=self.constraint_fns,
            sampler=self.sampler,
            cfg=self.cfg,
            rng=self.rng,
            # TODO: pass a single oracle(batch) callable instead of net/loss/fns
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
