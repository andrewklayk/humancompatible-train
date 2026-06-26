"""
Unbiased randomized-multilevel-Monte-Carlo estimate of the SCP search
direction (Facchinei & Kungurtsev, 2025, Eq. 23).

A single-sample SAA solve of the QP (3) is a *biased* estimate of the true
direction d(x), because d(·) is a nonlinear function of the sampled data. The
randomized-MLMC estimator removes that bias:

    N ~ Geometric on {0, 1, 2, ...},   p(N) = α (1-α)^N
    Δ_N = d(S_{2^{N+1}}) - ½ [ d(S_even) + d(S_odd) ]
    d̃(x) = d(X_single) + Δ_N / p(N)

where ``S_{2^{N+1}}`` is a fresh batch of 2^{N+1} samples and ``S_even`` /
``S_odd`` are its index-parity halves (2^N samples each).

Two details are essential for correctness:

* **Level support.** ``numpy``'s ``geometric(α)`` is supported on {1, 2, ...}
  with pmf α(1-α)^{N-1}. We draw ``geometric(α) - 1`` so that N lives on
  {0, 1, ...} with pmf α(1-α)^N -- this both matches the divisor ``p(N)`` and
  includes the level-0 increment (1 → 2 samples) that an off-by-one would drop.
* **Antithetic coupling.** The coarse solves use the even/odd halves of the
  *same* fine batch (not independent draws), so Var(Δ_N) decays geometrically
  and outpaces the 1/p(N) blow-up, giving the finite variance required by
  Assumption 2.2.
"""

import numpy as np

from .oracle import evaluate_oracle
from .subproblems import compute_kappa, solve_subproblem


def mlmc_direction(net, objective_fn, constraint_fns, sampler, cfg, rng):
    """Compute one unbiased step estimate d̃(x) (Eq. 23).

    Returns ``(d_tilde, info)`` where ``d_tilde`` is an (n,) ndarray and
    ``info`` holds per-step diagnostics.
    """
    # Geometric level on {0, 1, 2, ...} with p(N) = α (1-α)^N.
    N = int(rng.geometric(cfg.alpha)) - 1
    p_N = cfg.alpha * (1.0 - cfg.alpha) ** N
    base = cfg.base_batch
    J_base = base                # level-0 (base) sample count
    J_fine = base * 2 ** (N + 1)  # fine batch; antithetic halves are base * 2^N each

    def solve(batch):
        grad_f, cval, cgrad = evaluate_oracle(net, objective_fn, constraint_fns, batch)
        kappa = compute_kappa(cval, cgrad, cfg.lambda_, cfg.rho)
        d = solve_subproblem(grad_f, cval, cgrad, kappa, cfg.beta, cfg.tau, cfg.qp_solver)
        return d, cval, kappa

    # Base term: level-0 direction d(x; X_base) on J_base fresh samples.
    d_single, _, _ = solve(sampler.draw(J_base, rng))

    # Level-N increment from a fresh fine batch and its antithetic halves.
    fine = sampler.draw(J_fine, rng)
    even, odd = sampler.split_even_odd(fine)
    d_fine, cval, kappa = solve(fine)
    d_even, _, _ = solve(even)
    d_odd, _, _ = solve(odd)

    delta_N = d_fine - 0.5 * (d_even + d_odd)
    d_tilde = d_single + delta_N / p_N

    info = {
        "N": N,
        "p_N": p_N,
        "kappa": kappa,
        "constraint_values": cval,
        "n_samples": J_base + J_fine,  # distinct samples (even/odd reuse the fine batch)
        "fine_batch": J_fine,
    }
    return d_tilde, info
