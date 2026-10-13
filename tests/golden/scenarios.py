"""
Characterization ("golden") scenarios for the dual optimizers.

These exist to protect the extraction of :class:`DualOptimizer` (the shared base
class) from silently changing any algorithm. Each scenario drives one optimizer
through a fixed number of steps on a fixed, seeded toy problem and records the
full trajectory of every piece of mutable state -- duals, penalties, momentum
buffers, adaptive hyperparameters, the returned Lagrangian, and the primal
iterate.

``generate.py`` writes those trajectories to ``tests/golden/*.pt``;
``tests/test_golden.py`` re-runs the scenarios and asserts *bitwise* equality
against the stored ones. Goldens are generated from the pre-refactor code, so
any behavioural drift shows up as an exact-equality failure at a known step.

The toy problem deliberately uses a real autograd graph (constraints are
functions of the model output, not constants) so that the Lagrangian's backward
path is exercised too, and a plain SGD primal step so the recorded numbers are
reproducible.
"""

import torch

from humancompatible.train.dual_optim import ALM, PBM, iALM, nuPI

N_STEPS = 40
EPOCH_LENGTH = 8  # PBM annealing: N_STEPS / EPOCH_LENGTH = 5 epochs
PRIMAL_LR = 0.05

# name -> (factory, m, mode)
SCENARIOS = {}


def _scenario(name, m, mode="forward_update"):
    """Register a scenario. ``factory(epoch_length) -> dual optimizer``."""

    def deco(factory):
        SCENARIOS[name] = (factory, m, mode)
        return factory

    return deco


# --------------------------------------------------------------------------- #
# the toy problem
# --------------------------------------------------------------------------- #


def _problem(m, seed=0):
    """A seeded least-squares problem with ``m`` differentiable constraints.

    Returns ``(params, forward_fn)`` where ``forward_fn() -> (loss, constraints)``.
    The constraint offsets are chosen so that the constraint vector at the
    initial iterate is exactly ``linspace(-0.4, 0.4, m)`` -- a well-scaled mix of
    satisfied and violated constraints, which keeps every penalty/barrier branch
    reachable without any of them blowing up.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(8, 4, generator=g)
    y = torch.randn(8, generator=g)
    A = torch.randn(m, 8, generator=g)

    w = torch.randn(4, generator=g).requires_grad_(True)
    b = torch.zeros((), requires_grad=True)

    def raw_constraints():
        out = X @ w + b
        return A @ out / 8.0

    with torch.no_grad():
        offsets = raw_constraints() - torch.linspace(-0.4, 0.4, m)

    def forward_fn():
        out = X @ w + b
        loss = torch.mean((out - y) ** 2)
        constraints = A @ out / 8.0 - offsets
        return loss, constraints

    return [w, b], forward_fn


# --------------------------------------------------------------------------- #
# state capture
# --------------------------------------------------------------------------- #


def _snapshot(opt, params, lagrangian):
    """Capture every piece of mutable state we care about, as flat tensors."""
    snap = {
        "duals": opt.duals.detach().clone(),
        "lagrangian": lagrangian.detach().clone().reshape(()),
        "params": torch.cat([p.detach().reshape(-1).clone() for p in params]),
    }

    penalties = getattr(opt, "penalties", None)
    if penalties is not None:
        snap["penalties"] = penalties.detach().clone()

    buffers = [
        g["momentum_buffer"] for g in opt.param_groups if "momentum_buffer" in g
    ]
    if buffers:
        snap["buffers"] = torch.cat([t.detach().reshape(-1).clone() for t in buffers])

    # iALM mutates beta in place; record it so the sigma schedule is pinned.
    betas = [
        g["beta"]
        for g in opt.param_groups
        if isinstance(g.get("beta"), torch.Tensor)
    ]
    if betas:
        snap["betas"] = torch.stack([t.detach().reshape(()).clone() for t in betas])

    return snap


def run_scenario(name):
    """Run one scenario and return ``{key: stacked trajectory tensor}``."""
    factory, m, mode = SCENARIOS[name]

    torch.manual_seed(0)
    params, forward_fn = _problem(m)
    dual = factory(EPOCH_LENGTH)
    primal = torch.optim.SGD(params, lr=PRIMAL_LR)

    # Pin the constructor's own output too: without this the first recorded frame
    # is already post-step, so init_duals / init_penalties / range defaults would
    # only be covered indirectly.
    initial = {
        "initial_duals": dual.duals.detach().clone(),
        "initial_params": torch.cat([p.detach().reshape(-1).clone() for p in params]),
    }
    if getattr(dual, "penalties", None) is not None:
        initial["initial_penalties"] = dual.penalties.detach().clone()

    frames = []
    for _ in range(N_STEPS):
        primal.zero_grad()
        loss, constraints = forward_fn()

        if mode == "forward_update":
            lagrangian = dual.forward_update(loss, constraints)
            lagrangian.backward()
            primal.step()
        elif mode == "split":
            # The dual update must come after backward(): forward() builds the
            # Lagrangian from the *current* duals, and autograd saves that tensor
            # for the matmul's backward, so updating them first bumps its version
            # and invalidates the graph.
            lagrangian = dual.forward(loss, constraints)
            lagrangian.backward()
            primal.step()
            dual.update(constraints.detach())
        else:
            raise ValueError(f"unknown mode {mode!r}")

        frames.append(_snapshot(dual, params, lagrangian))

    traj = {k: torch.stack([f[k] for f in frames]) for k in frames[0]}
    traj.update(initial)
    return traj


# --------------------------------------------------------------------------- #
# ALM
# --------------------------------------------------------------------------- #


@_scenario("alm_basic", m=3)
def _(epoch_length):
    return ALM(m=3, lr=0.1, penalty=1.0)


@_scenario("alm_basic_split", m=3, mode="split")
def _(epoch_length):
    return ALM(m=3, lr=0.1, penalty=1.0)


@_scenario("alm_momentum", m=3)
def _(epoch_length):
    # dampening left unset -> defaults to momentum (EMA)
    return ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9)


@_scenario("alm_momentum_dampening", m=3)
def _(epoch_length):
    return ALM(m=3, lr=0.1, penalty=1.0, momentum=0.9, dampening=0.5)


@_scenario("alm_ineq", m=4)
def _(epoch_length):
    return ALM(m=4, lr=0.2, penalty=1.0, is_ineq=True)


@_scenario("alm_ineq_restart", m=4)
def _(epoch_length):
    return ALM(m=4, lr=0.2, penalty=1.0, is_ineq=True, restart=True)


@_scenario("alm_clamped", m=3)
def _(epoch_length):
    # tight range so the clamp is active for most of the run
    return ALM(m=3, lr=0.5, penalty=1.0, dual_range=(-0.05, 0.05))


@_scenario("alm_zero_penalty", m=3)
def _(epoch_length):
    return ALM(m=3, lr=0.1, penalty=0.0)


@_scenario("alm_init_duals", m=3)
def _(epoch_length):
    return ALM(m=3, lr=0.1, penalty=1.0, init_duals=0.7)


@_scenario("alm_two_groups", m=5)
def _(epoch_length):
    opt = ALM(m=2, lr=0.1, penalty=1.0, momentum=0.9)
    opt.add_constraint_group(m=3, lr=0.2, is_ineq=True)
    return opt


# --------------------------------------------------------------------------- #
# nuPI
# --------------------------------------------------------------------------- #


@_scenario("nupi_basic", m=3)
def _(epoch_length):
    return nuPI(m=3, nu=0.01, penalty=1.0, ki=0.01, kp=1.0)


@_scenario("nupi_basic_split", m=3, mode="split")
def _(epoch_length):
    return nuPI(m=3, nu=0.01, penalty=1.0, ki=0.01, kp=1.0)


@_scenario("nupi_high_nu", m=3)
def _(epoch_length):
    return nuPI(m=3, nu=0.99, penalty=1.0, ki=0.05, kp=1.0)


@_scenario("nupi_zero_nu", m=3)
def _(epoch_length):
    # nu=0 is the variant used in the sparsity experiments of the reference
    return nuPI(m=3, nu=0.0, penalty=1.0, ki=0.05, kp=0.5)


@_scenario("nupi_ineq", m=4)
def _(epoch_length):
    return nuPI(m=4, nu=0.5, penalty=1.0, ki=0.02, kp=1.0, is_ineq=True)


@_scenario("nupi_clamped", m=3)
def _(epoch_length):
    return nuPI(m=3, nu=0.5, penalty=1.0, ki=0.5, kp=2.0, dual_range=(-0.05, 0.05))


@_scenario("nupi_two_groups", m=5)
def _(epoch_length):
    opt = nuPI(m=2, nu=0.01, penalty=1.0, ki=0.01, kp=1.0)
    opt.add_constraint_group(m=3, nu=0.9, ki=0.05, kp=0.5, is_ineq=True)
    return opt


# --------------------------------------------------------------------------- #
# iALM
# --------------------------------------------------------------------------- #


@_scenario("ialm_basic", m=3)
def _(epoch_length):
    return iALM(m=3, beta=1.0, sigma=1.0, gamma=1.0)


@_scenario("ialm_basic_split", m=3, mode="split")
def _(epoch_length):
    return iALM(m=3, beta=1.0, sigma=1.0, gamma=1.0)


@_scenario("ialm_growing_beta", m=3)
def _(epoch_length):
    # sigma > 1 -> geometric penalty schedule
    return iALM(m=3, beta=0.5, sigma=1.05, gamma=1.0)


@_scenario("ialm_small_gamma", m=3)
def _(epoch_length):
    # gamma small -> the min(beta, gamma/||c||) branch binds
    return iALM(m=3, beta=2.0, sigma=1.0, gamma=0.01)


@_scenario("ialm_momentum", m=3)
def _(epoch_length):
    return iALM(m=3, beta=1.0, sigma=1.0, gamma=1.0, momentum=0.9)


@_scenario("ialm_ineq", m=4)
def _(epoch_length):
    return iALM(m=4, beta=1.0, sigma=1.02, gamma=1.0, is_ineq=True)


@_scenario("ialm_two_groups", m=5)
def _(epoch_length):
    opt = iALM(m=2, beta=1.0, sigma=1.0, gamma=1.0)
    opt.add_constraint_group(m=3, beta=0.5, sigma=1.1, gamma=0.5, is_ineq=True)
    return opt


# --------------------------------------------------------------------------- #
# PBM
# --------------------------------------------------------------------------- #


@_scenario("pbm_dimin_adapt", m=3)
def _(epoch_length):
    return PBM(m=3, penalty_update="dimin_adapt", epoch_length=epoch_length)


@_scenario("pbm_dimin_adapt_split", m=3, mode="split")
def _(epoch_length):
    return PBM(m=3, penalty_update="dimin_adapt", epoch_length=epoch_length)


@_scenario("pbm_dimin", m=3)
def _(epoch_length):
    # Wide range + high initial penalty so the geometric decay stays *inside* the
    # range for all N_STEPS. With the default (0.1, 1.0) range the penalties hit
    # the clamp floor on the first step and the trajectory pins nothing;
    # pbm_dimin_saturating below covers that regime deliberately.
    return PBM(
        m=3,
        penalty_update="dimin",
        penalty_mult=0.9,
        init_penalties=50.0,
        penalty_range=(1e-6, 100.0),
        epoch_length=epoch_length,
    )


@_scenario("pbm_dimin_saturating", m=3)
def _(epoch_length):
    # The default narrow range: penalties collapse to the clamp floor immediately
    # and stay there. Saturation is real behaviour and worth pinning separately.
    return PBM(m=3, penalty_update="dimin", epoch_length=epoch_length)


@_scenario("pbm_dimin_dual", m=3)
def _(epoch_length):
    return PBM(
        m=3,
        penalty_update="dimin_dual",
        penalty_mult=0.9,
        init_penalties=50.0,
        penalty_range=(1e-6, 100.0),
        epoch_length=epoch_length,
    )


@_scenario("pbm_const", m=3)
def _(epoch_length):
    return PBM(m=3, penalty_update="const", epoch_length=epoch_length)


@_scenario("pbm_aimd", m=3)
def _(epoch_length):
    return PBM(m=3, penalty_update="aimd", epoch_length=epoch_length)


@_scenario("pbm_alm_penalty", m=3)
def _(epoch_length):
    # the ablation knob that reduces SPBM to ALM; needs a wide penalty range
    return PBM(
        m=3,
        penalty_update="alm",
        rho=2.0,
        penalty_range=(0.1, 100.0),
        epoch_length=epoch_length,
    )


@_scenario("pbm_quad_recipr", m=3)
def _(epoch_length):
    return PBM(m=3, pbf="quadratic_reciprocal", epoch_length=epoch_length)


@_scenario("pbm_no_annealing", m=3)
def _(epoch_length):
    return PBM(
        m=3,
        gamma=0.5,
        gamma_annealing=False,
        penalty_annealing=False,
    )


@_scenario("pbm_pupl2", m=3)
def _(epoch_length):
    # several primal steps per dual step
    return PBM(m=3, primal_update_process_length=2, epoch_length=epoch_length)


@_scenario("pbm_delta", m=3)
def _(epoch_length):
    return PBM(m=3, delta=2.0, penalty_mult=0.7, epoch_length=epoch_length)


@_scenario("pbm_init_values", m=3)
def _(epoch_length):
    return PBM(
        m=3,
        init_duals=0.5,
        init_penalties=0.8,
        dual_range=(1e-4, 10.0),
        epoch_length=epoch_length,
    )


@_scenario("pbm_two_groups", m=5)
def _(epoch_length):
    opt = PBM(
        m=2,
        penalty_update="dimin_adapt",
        init_penalties=50.0,
        penalty_range=(1e-6, 100.0),
        epoch_length=epoch_length,
    )
    opt.add_constraint_group(
        m=3,
        penalty_update="dimin",
        penalty_mult=0.9,
        init_penalties=20.0,
        pbf="quadratic_reciprocal",
    )
    return opt
