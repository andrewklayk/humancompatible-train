"""
E0a — mathematical faithfulness of the dual optimizers.

The question is whether each implementation in
``humancompatible.train.dual_optim`` encodes the surrogate and multiplier update
its reference specifies. That is an **algebraic** property, and this experiment
tests it algebraically. Convergence is a separate question — it depends on step
sizes, conditioning and tuning — and is reported here in one untuned
configuration per method, not as a ranking.

An earlier version of E0a tested convergence and used it as a proxy for
faithfulness. That produced predictions naming particular methods at particular
hyperparameters, four unrelated tolerances, and one failure that turned out to be
an artifact of two of those thresholds being mismatched by 10^4. See the
superseded-predictions note in paper/README.md.

Four categories of claim, in decreasing strength:

**F — fixed-point consistency.** At a KKT point ``(x*, y*)``, a faithful method
must be stationary: one ``forward_update`` from ``(x*, y*)`` must leave the
multipliers where they are, and ``grad_x`` of the surrogate must vanish. No step
size, no tuning, no iteration. This is the strongest test available and it is
sharp: switching ``ALM``'s quadratic term back to raw ``c`` on inequality data
takes ``grad_x`` from 1.8e-15 to 4.7 on ``qp_inactive``, so F is a live regression
guard for the ``[c]_+`` fix. It also requires the non-negativity clamp, and detects
the loss of *either* ingredient — it does not isolate one of them.

**R — exact reductions between methods.** Three of the four classes reduce to
``ALM`` under specific settings, which tests them against an independent
implementation rather than against a tolerance:

======  ==========================================================  ================================
R1      ``nuPI(kp=0, ki=g)``            == ``ALM(lr=g, rho=0)``     no preconditions
R2      ``iALM(beta, sigma=1, gamma>>)``== ``ALM(lr=b, rho=b)``     ``gamma >= beta*||c||``
R3      ``PBM(penalty_update="alm")``   == ``ALM((1-g)/r, 1/r)``    ``p0 = y0*r``; ``c/p >= -0.5``
======  ==========================================================  ================================

R3 is the strongest single assertion in the experiment: with ``p = y*r`` the
quadratic-logarithmic barrier ``sum_i y_i p_i phi(c_i/p_i)`` collapses to
``y'c + ||c||^2/(2r)``, so one comparison validates PBM's barrier algebra, its
penalty update and its dual rule together. Note ``PBM`` snapshots multipliers
*pre*-update and ``ALM`` *post*-update, so the comparison is ``PBM.forward_update``
against ``ALM.forward`` **then** ``ALM.update``. Each reduction is checked both for
a single step and along a trajectory, reporting the step at which a precondition
first breaks — that boundary is informative in itself.

**I — invariances the mathematics requires.**

* I1 ``c -> alpha*c`` implies ``y* -> y*/alpha``. Catches an absolute constraint
  scale leaking into a dual rule.
* I2 an equality ``h = 0`` expressed as ``h <= 0, -h <= 0`` recovers ``x*`` and
  ``y+ - y- = y_eq``. The package's problem statement relies on this reduction and
  nothing else tests it. Individual ``y+``/``y-`` are not determined, so only the
  difference is asserted.

**C — convergence, one configuration per method.** The primal step is derived as
``1/(L_f + rho*||J||^2)`` and the dual step is ``1/||J||^2`` for every method on
every problem, so nothing here is tuned. Expectations are stated per problem and
conditioned on a *structural predicate* of the method rather than on its name:

* ``qp_active`` — strongly convex, all constraints active, ``y* > 0``, LICQ and
  strict complementarity, so the KKT point is unique. **Every** method must
  converge; a failure is a defect.
* ``qp_inactive`` — adds exact zeros in ``y*``. Every method whose declared
  ``lower_bound`` is 0 reaches ``y*_i = 0`` exactly; one with ``lower_bound > 0``
  cannot, and its error is bounded below by that bound.
* ``svm_iris`` — 96 of 100 multipliers are zero, so the same ``lower_bound`` rule as
  ``qp_inactive`` applies. The objective Hessian is **singular** in the bias, which
  turns out to slow convergence rather than prevent it: ``ALM(rho=0)`` reaches 9.1e-14
  given the derived budget. With ``m = 100`` and ``n = 5``, ``J'`` has a
  95-dimensional null space, so stationarity alone does not pin ``y`` — only
  complementarity does, which is why the residual includes ``||y - y*||``.
* ``qp_nonconvex`` — no convergence theory exists for a fixed-penalty Lagrangian on
  an indefinite objective. **Nothing is claimed**; outcomes are counted.

**O — observations, reported but not gated.** The ``qp_nonconvex`` outcome counts,
framed as limits of applicability rather than as defects.

Tolerances, one rule per category rather than four ad-hoc constants: F, R and I are
algebraic identities, so they assert bitwise where the operations are literally
identical and ``64*eps*scale`` otherwise — anything above rounding is a bug. C
reports a relative KKT residual and states the value achieved.

Usage::

    python paper/e0/a_multipliers.py            # everything
    python paper/e0/a_multipliers.py --quick    # F, R, I only (algebra, seconds)
    python paper/e0/a_multipliers.py --check    # exit non-zero on a failed claim
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from humancompatible.train.dual_optim import ALM, PBM, DualOptimizer, iALM, nuPI
from paper._harness import (
    Checks,
    figure,
    main_exit,
    save_figure,
    set_seed,
    use_float64,
    write_csv,
    write_table,
)
from paper.problems import Problem
from paper.problems.qp import (
    qp_active,
    qp_equality,
    qp_equality_reduced,
    qp_inactive,
    qp_nonconvex,
)
from paper.problems.svm import svm_iris

EXPERIMENT = "e0a"

# Algebraic tolerance: these are identities, so the only admissible error is
# floating-point rounding. 64 eps leaves room for a few dozen dependent
# operations without leaving room for a wrong formula.
ALGEBRAIC = 64 * float(np.finfo(np.float64).eps)
# Convergence tolerance, on the *relative* KKT residual.
KKT_TOLERANCE = 1e-6
SCALE_ALPHA = 7.0          # for I1; not a round number, so a missing factor shows

# Iteration budget for C, in units of "iterations x dual step". A *fixed* iteration
# count is not comparable across problems: with the dual step derived as
# 1/||J||^2, the number of dual updates needed to carry y from 0 to y* scales like
# ||y*||*||J||^2, so a count that converges on qp_active (||J||^2 = 19) is 16x too
# small for svm_iris (||J||^2 = 305). Measured on svm_iris at its derived step:
# 1.6e-03 relative KKT after 20k iterations, 4.7e-06 after 50k, 2.8e-10 after 100k,
# 4.6e-14 after 200k. 700 puts every problem comfortably past convergence.
DUAL_STEP_BUDGET = 700.0


def _iterations(problem: Problem) -> int:
    """``budget / dual_step``, i.e. equal progress rather than equal iterations."""
    return int(np.ceil(DUAL_STEP_BUDGET * problem.jac_norm_sq))


# --------------------------------------------------------------------------- #
# method configurations
# --------------------------------------------------------------------------- #


@dataclass
class Method:
    """One method at one configuration.

    :param build: ``build(problem) -> DualOptimizer``, already carrying this
        configuration's dual step.
    :param penalty_coefficient: The quadratic coefficient, used only to derive the
        primal step size.
    :param seed_buffer: True when the method carries an error buffer that must be
        initialised to ``c(x*)`` for the fixed-point test to be meaningful (nuPI's
        PI recursion is stationary only once its buffer has converged).
    :param equality_via_two_sided: Whether ``h <= 0, -h <= 0`` is a documented route
        to an equality for this method (I2). False for ``PBM``, and structurally so:
        the two-sided form has no strictly feasible interior, since at any ``x`` one
        of each pair is violated. PBM's multiplicative update
        ``y <- gamma*y + (1-gamma)*y*phi'(c/p)`` then grows whichever side is
        violated and alternates — measured ping-ponging between its 1e-4 floor and
        its 100 ceiling. Its documented route is the threshold ``|h| <= tau``, which
        keeps an interior but poses a *different* problem with different
        multipliers, so it is not comparable here and is not attempted.
    """

    label: str
    build: Callable[[Problem], DualOptimizer]
    penalty_coefficient: float
    has_curvature: bool          # a quadratic term in the surrogate
    has_damping: bool            # a proportional/derivative term in the dual rule
    seed_buffer: bool = False
    equality_via_two_sided: bool = True
    style: dict = field(default_factory=dict)

    def dual_step(self, problem: Problem) -> float:
        """``1/||J||^2`` — dimensionally correct and identical for every method."""
        return 1.0 / problem.jac_norm_sq


def _methods() -> list[Method]:
    """One configuration per method. No sweep: nothing here is tuned."""
    def alm(rho, color, ls, marker):
        return Method(
            f"ALM (rho={rho:g})",
            lambda p, rho=rho: ALM(m=p.m, lr=1.0 / p.jac_norm_sq, penalty=rho,
                                   init_duals=0.0, is_ineq=True),
            penalty_coefficient=rho, has_curvature=rho > 0, has_damping=False,
            style={"color": color, "ls": ls, "marker": marker},
        )

    def pi(rho, color, ls):
        # kp = ki keeps one knob; nu stays at the reference default.
        return Method(
            f"nuPI (rho={rho:g})",
            lambda p, rho=rho: nuPI(m=p.m, ki=1.0 / p.jac_norm_sq,
                                    kp=1.0 / p.jac_norm_sq, nu=0.01, penalty=rho,
                                    init_duals=0.0, is_ineq=True),
            penalty_coefficient=rho, has_curvature=rho > 0, has_damping=True,
            seed_buffer=True,
            style={"color": color, "ls": ls, "marker": "v"},
        )

    return [
        alm(0.0, "#7FB2D0", "--", "o"),
        alm(1.0, "#2E86AB", "-", "s"),
        pi(0.0, "#E0A458", "--"),
        pi(1.0, "#A8621B", "-"),
        Method(
            "iALM",
            # beta is simultaneously the quadratic coefficient and the dual step,
            # so it takes the same 1/||J||^2 value; gamma is left at its default so
            # the published safeguard stays active.
            lambda p: iALM(m=p.m, beta=1.0 / p.jac_norm_sq, sigma=1.0, gamma=1.0,
                           init_duals=0.0, is_ineq=True),
            penalty_coefficient=1.0, has_curvature=True, has_damping=False,
            style={"color": "#5BC0BE", "ls": ":", "marker": "^"},
        ),
        Method(
            "PBM",
            lambda p: PBM(m=p.m, gamma=0.5, penalty_mult=0.1, delta=1.0,
                          penalty_update="dimin_adapt", gamma_annealing=False,
                          penalty_annealing=False),
            # phi'' <= 1 for the quadratic-logarithmic barrier, so the curvature is
            # bounded by max_i (y_i/p_i)*||J||^2; 1.0 holds at these solutions.
            penalty_coefficient=1.0, has_curvature=True, has_damping=False,
            equality_via_two_sided=False,      # see the field's docstring
            style={"color": "#D1495B", "ls": "-", "marker": "D"},
        ),
    ]


def _reference_problems() -> list[Problem]:
    """Problems with an exact ``(x*, y*)``, so F and C can both run."""
    return [qp_active(), qp_inactive(), svm_iris()]


def _all_problems() -> list[Problem]:
    return _reference_problems() + [qp_nonconvex()]


# --------------------------------------------------------------------------- #
# shared machinery
# --------------------------------------------------------------------------- #


def _seed_buffer(optimizer: DualOptimizer, c: torch.Tensor) -> None:
    """Put nuPI's error buffer at ``c``, which is where a fixed point needs it.

    The PI recursion is ``y += (ki + kp(1-nu))c - kp(1-nu)xi``; only once
    ``xi == c`` does that collapse to ``y += ki*c``, so a fixed-point test run from
    a zero buffer would report a spurious deviation of ``kp(1-nu)*c``.
    """
    for group in optimizer.param_groups:
        buffer = group.get("momentum_buffer")
        if buffer is not None:
            buffer.copy_(c.detach())


def _metrics(problem: Problem, x: torch.Tensor, duals: torch.Tensor) -> dict:
    """KKT diagnostics at ``(x, y)``, exact — no minibatch noise anywhere in E0a."""
    f = problem.objective([x])
    c = problem.constraints([x])
    # grad_x (f + y'c) = grad f + J'y in one backward pass, never forming J.
    (gradient,) = torch.autograd.grad(f + duals @ c, x)
    f, c = f.detach(), c.detach()

    row = {
        "f": float(f),
        "violation": float(c.clamp(min=0.0).max()),
        "stationarity": float(gradient.abs().max()),
        "complementarity": float((duals * c).abs().max()),
        "dual_min": float(duals.min()),
    }
    if problem.f_star is not None:
        row["f_gap"] = float(f) - problem.f_star
    if problem.y_star is not None:
        y_star = torch.as_tensor(problem.y_star, dtype=duals.dtype)
        error = duals - y_star
        row["y_inf"] = float(error.abs().max())
        row["y_rel"] = float(error.norm() / max(1e-30, float(y_star.norm())))
    return row


def _relative_kkt(problem: Problem, row: dict) -> float:
    """A single scale-free residual, so one tolerance suffices for every problem."""
    scale = 1.0 + (0.0 if problem.y_star is None
                   else float(np.abs(problem.y_star).max()))
    parts = [row["stationarity"] / scale, row["violation"]]
    if "y_inf" in row:
        parts.append(row["y_inf"] / scale)
    return max(parts)


def _status(final: dict) -> str:
    """``diverged`` / ``bounded`` / ``solved``, on the same residual the gates use.

    ``bounded`` needs its own name: on ``qp_nonconvex`` a method can stay finite at
    violation 36, which a two-way diverged/ok split would call "ok".

    This must read the *relative KKT* residual, not ``max(stationarity, violation)``.
    On ``svm_iris`` ``m = 100`` and ``n = 5``, so ``J'`` has a 95-dimensional null
    space and stationarity alone does not determine ``y`` — only stationarity plus
    complementarity does. A method can therefore be feasible and stationary while its
    multipliers are wrong by 0.25, and the narrower residual would call that "solved".
    """
    if final.get("diverged"):
        return "diverged"
    residual = final.get("relative KKT", float("inf"))
    return "solved" if residual <= KKT_TOLERANCE else "bounded"


# --------------------------------------------------------------------------- #
# F — fixed-point consistency
# --------------------------------------------------------------------------- #


def fixed_point(problem: Problem, method: Method) -> dict:
    """One ``forward_update`` from ``(x*, y*)``.

    :return: The surrogate's ``grad_x`` norm, the multiplier drift, and the
        deviation the method's own declared ``lower_bound`` makes unavoidable.
    """
    set_seed(0)
    x = torch.nn.Parameter(torch.as_tensor(problem.x_star,
                                           dtype=torch.get_default_dtype()))
    y_star = torch.as_tensor(problem.y_star, dtype=torch.get_default_dtype())
    c_star = problem.constraints([x]).detach()

    optimizer = method.build(problem)
    lower = optimizer.param_groups[0].get("lower_bound")
    # A method whose multipliers must stay strictly positive cannot be started at
    # an exact zero, so it is started as close as its own bound allows -- and that
    # distance is the deviation we then expect to see, rather than a failure.
    start = y_star if lower is None else y_star.clamp(min=lower)
    expected = float((start - y_star).abs().max())
    for group in optimizer.param_groups:
        group["params"][0].data.copy_(start)
    if method.seed_buffer:
        _seed_buffer(optimizer, c_star)

    surrogate = optimizer.forward_update(problem.objective([x]),
                                         problem.constraints([x]))
    (gradient,) = torch.autograd.grad(surrogate, x)
    drift = float((optimizer.duals.detach() - y_star).abs().max())

    scale = 1.0 + float(np.abs(problem.y_star).max())
    # A method that cannot start at y* also cannot have a vanishing surrogate
    # gradient there: the offset enters grad_x through the Jacobian, to first
    # order as ||J|| * ||y - y*||. Derive that allowance rather than loosening a
    # tolerance until PBM fits under it.
    transmitted = expected * float(np.sqrt(problem.jac_norm_sq))
    return {
        "problem": problem.name,
        "method": method.label,
        "grad_x": float(gradient.abs().max()),
        "dual drift": drift,
        "lower_bound": lower,
        "unavoidable deviation": expected,
        # What the method is answerable for, once its own declared bound is allowed.
        "excess drift": max(0.0, drift - expected),
        "tolerance": ALGEBRAIC * scale,
        # ALGEBRAIC*1e3 because grad_x accumulates the problem's data scale through
        # J; 10x the transmitted offset because that estimate is first-order.
        "grad_x tolerance": max(ALGEBRAIC * 1e3, 10.0 * transmitted),
    }


def register_fixed_point(checks: Checks, rows: list[dict]) -> None:
    for row in rows:
        tol, grad_tol = row["tolerance"], row["grad_x tolerance"]
        checks.expect(
            row["grad_x"] <= grad_tol and row["excess drift"] <= tol,
            f"F: {row['method']} is a fixed point at (x*, y*) on {row['problem']}",
            f"grad_x {row['grad_x']:.3e} (tol {grad_tol:.1e}); excess drift "
            f"{row['excess drift']:.3e} (tol {tol:.1e}); unavoidable "
            f"{row['unavoidable deviation']:.3e} from lower_bound="
            f"{row['lower_bound']}",
        )


def register_fixed_point_is_sharp(checks: Checks, problem: Problem) -> None:
    """F must FAIL when the quadratic term is put back on raw ``c``.

    Without this the whole category could be passing vacuously. ``is_ineq=False``
    is how the pre-B.1 behaviour is reachable through the public API; note it also
    drops the non-negativity clamp, so this shows F detects the loss of either
    required ingredient, not that it isolates the penalty.
    """
    set_seed(0)
    x = torch.nn.Parameter(torch.as_tensor(problem.x_star,
                                           dtype=torch.get_default_dtype()))
    y_star = torch.as_tensor(problem.y_star, dtype=torch.get_default_dtype())
    optimizer = ALM(m=problem.m, lr=1.0 / problem.jac_norm_sq, penalty=1.0,
                    init_duals=y_star.clone(), is_ineq=False)
    surrogate = optimizer.forward_update(problem.objective([x]),
                                         problem.constraints([x]))
    (gradient,) = torch.autograd.grad(surrogate, x)
    broken = float(gradient.abs().max())
    drift = float((optimizer.duals.detach() - y_star).abs().max())
    checks.expect(
        broken > 1e-3 and drift > 1e-3,
        "F is sharp: with the quadratic term on raw c and no non-negativity clamp, "
        f"the fixed point breaks on {problem.name} (regression guard for the [c]+ fix)",
        f"grad_x {broken:.3e}, dual drift {drift:.3e} — both must be large",
    )


# --------------------------------------------------------------------------- #
# R — exact reductions to ALM
# --------------------------------------------------------------------------- #


def _reduction_pairs(problem: Problem):
    """``(name, build_reduced, build_alm, alm_split, precondition, bitwise, note)``.

    ``precondition(reduced, c)`` reports whether the algebra that makes the
    reduction exact still applies at the current iterate. A reduction is only
    asserted over the steps where its preconditions hold; the step at which one
    first breaks is reported, since that boundary is the useful information.
    """
    g = 1.0 / problem.jac_norm_sq          # the shared dual step
    m = problem.m
    gamma_p = 0.4
    y0 = 0.75                              # strictly positive: PBM needs it
    span = (1e-10, 1e10)

    # R3 lives on quad_log's quadratic branch, which needs c/p >= -0.5. Since
    # p_0 = y0*rho_p, pick rho_p so that holds at the starting iterate with margin
    # -- otherwise the reduction is being tested where it is not claimed to apply.
    x0 = problem.make_params()[0]
    c0 = float(problem.constraints([x0]).detach().abs().max())
    rho_p = max(3.0, 4.0 * c0 / y0)

    def r2_ok(reduced, c):
        # min(beta, gamma/||c||) must select beta for the step to equal ALM's lr.
        beta = float(reduced.param_groups[0]["beta"])
        return 1e12 >= beta * float(c.norm())

    def r3_ok(reduced, c):
        # quad_log is the quadratic branch only for t >= -0.5, and neither the
        # duals nor the penalties may be sitting on a safeguarding clamp.
        t = (c.detach() / reduced.penalties).min()
        duals, pens = reduced.duals.detach(), reduced.penalties
        inside = (duals.min() > span[0]) and (duals.max() < span[1]) \
            and (pens.min() > span[0]) and (pens.max() < span[1])
        return bool(t >= -0.5) and bool(inside)

    return [
        (
            "R1  nuPI(kp=0) == ALM(rho=0)",
            lambda: nuPI(m=m, ki=g, kp=0.0, nu=0.01, penalty=0.0,
                         init_duals=y0, is_ineq=True),
            lambda: ALM(m=m, lr=g, penalty=0.0, init_duals=y0, is_ineq=True),
            False,
            lambda reduced, c: True,
            True,      # duals.add_(c, alpha=ki) vs add_(c, alpha=lr): same op
            "unconditional",
        ),
        (
            "R2  iALM(sigma=1, gamma>>) == ALM(lr=beta, rho=beta)",
            lambda: iALM(m=m, beta=g, sigma=1.0, gamma=1e12,
                         init_duals=y0, is_ineq=True),
            lambda: ALM(m=m, lr=g, penalty=g, init_duals=y0, is_ineq=True),
            False,
            r2_ok,
            True,      # both reduce to a single add_(c, alpha=beta)
            "requires gamma >= beta*||c||, i.e. the safeguard does not bind",
        ),
        (
            "R3  PBM(penalty_update='alm') == ALM((1-g)/r, 1/r)",
            lambda: PBM(m=m, gamma=gamma_p, penalty_update="alm", rho=rho_p,
                        init_duals=y0, init_penalties=y0 * rho_p,
                        dual_range=span, penalty_range=span,
                        gamma_annealing=False, penalty_annealing=False),
            lambda: ALM(m=m, lr=(1.0 - gamma_p) / rho_p, penalty=1.0 / rho_p,
                        init_duals=y0, is_ineq=False, dual_range=span),
            # PBM snapshots pre-update, ALM post-update, so ALM must be driven
            # split (forward, then update) for the surrogates to line up.
            True,
            r3_ok,
            # y*(gamma + (1-gamma)(1 + c/p)) and y + lr*c are the same number by a
            # different route, so this one is held to rounding, not to bits.
            False,
            "requires c/p >= -0.5 (the quad_log branch) and no range clamps",
        ),
    ]


def reduction_trajectory(problem: Problem, build_a, build_b, b_split: bool,
                         precondition, steps: int = 200) -> dict:
    """Run two configurations in lockstep for as long as the algebra applies.

    Stops at the first iterate where ``precondition`` fails, so the assertion is
    made only where the reduction is claimed to hold.
    """
    set_seed(0)
    xa = problem.make_params()[0]
    set_seed(0)
    xb = problem.make_params()[0]
    a, b = build_a(), build_b()
    lr = problem.primal_step(1.0)
    opt_a = torch.optim.SGD([xa], lr=lr)
    opt_b = torch.optim.SGD([xb], lr=lr)

    worst_surrogate = worst_dual = 0.0
    ran = 0
    breach = None
    for step in range(steps):
        ca = problem.constraints([xa])
        if not precondition(a, ca):
            breach = step
            break
        sa = a.forward_update(problem.objective([xa]), ca)
        opt_a.zero_grad(); sa.backward(); opt_a.step()

        loss_b, c_b = problem.objective([xb]), problem.constraints([xb])
        if b_split:
            sb = b.forward(loss_b, c_b)
            opt_b.zero_grad(); sb.backward(); opt_b.step(); b.update(c_b)
        else:
            sb = b.forward_update(loss_b, c_b)
            opt_b.zero_grad(); sb.backward(); opt_b.step()

        worst_surrogate = max(worst_surrogate, abs(float(sa) - float(sb)))
        worst_dual = max(worst_dual,
                         float((a.duals.detach() - b.duals.detach()).abs().max()))
        ran = step + 1
    return {
        "steps requested": steps,
        "steps with preconditions holding": ran,
        "precondition first broke at step": breach,
        "max |surrogate difference|": worst_surrogate,
        "max |dual difference|": worst_dual,
    }


def register_reductions(checks: Checks, problems: list[Problem]) -> list[dict]:
    rows = []
    for problem in problems:
        for name, build_a, build_b, b_split, pre, bitwise, note in _reduction_pairs(
                problem):
            # single step from a common state
            set_seed(0)
            x = problem.make_params()[0]
            a, b = build_a(), build_b()
            sa = a.forward_update(problem.objective([x]), problem.constraints([x]))
            loss, c = problem.objective([x]), problem.constraints([x])
            if b_split:
                sb = b.forward(loss, c); b.update(c)
            else:
                sb = b.forward_update(loss, c)
            dual_gap = float((a.duals.detach() - b.duals.detach()).abs().max())
            one_step = {
                "surrogate difference": abs(float(sa) - float(sb)),
                "dual difference": dual_gap,
                "duals bitwise identical": torch.equal(a.duals.detach(),
                                                       b.duals.detach()),
                # Bitwise is only the right bar when the two updates execute the
                # *same* floating-point operations; R3 reaches the same number by a
                # different route, so it is held to rounding instead.
                "bar": "bitwise" if bitwise else f"<= {ALGEBRAIC * 1e3:.1e}",
            }
            traj = reduction_trajectory(problem, build_a, build_b, b_split, pre)
            rows.append({"problem": problem.name, "reduction": name,
                         "note": note, **one_step, **traj})

            duals_ok = (one_step["duals bitwise identical"] if bitwise
                        else dual_gap <= ALGEBRAIC * 1e3)
            checks.expect(
                one_step["surrogate difference"] <= ALGEBRAIC * 1e3 and duals_ok,
                f"{name} holds for one step on {problem.name} ({one_step['bar']})",
                f"surrogate difference {one_step['surrogate difference']:.3e}, "
                f"dual difference {dual_gap:.3e}, bitwise: "
                f"{one_step['duals bitwise identical']}",
            )
            ran = traj["steps with preconditions holding"]
            checks.expect(
                traj["max |dual difference|"] <= ALGEBRAIC * 1e3 and ran > 0,
                f"{name} holds for all {ran} steps on {problem.name} where its "
                f"preconditions apply",
                f"worst dual difference {traj['max |dual difference|']:.3e}; "
                f"preconditions broke at step "
                f"{traj['precondition first broke at step']} ({note})",
            )
    return rows


# --------------------------------------------------------------------------- #
# I — invariances
# --------------------------------------------------------------------------- #


def register_scaling_invariance(checks: Checks, methods: list[Method]) -> list[dict]:
    """I1: scaling the constraints must scale the multipliers, not move the KKT point."""
    rows = []
    for base in _reference_problems():
        scaled = base.scaled(SCALE_ALPHA)
        for method in methods:
            plain = fixed_point(base, method)
            lifted = fixed_point(scaled, method)
            rows.append({"problem": base.name, "method": method.label,
                         "alpha": SCALE_ALPHA,
                         "excess drift (plain)": plain["excess drift"],
                         "excess drift (scaled)": lifted["excess drift"]})
            checks.expect(
                lifted["excess drift"] <= lifted["tolerance"],
                f"I1: {method.label} is a fixed point at (x*, y*/{SCALE_ALPHA:g}) "
                f"for {base.name} with constraints scaled by {SCALE_ALPHA:g}",
                f"excess drift {lifted['excess drift']:.3e} "
                f"(unscaled {plain['excess drift']:.3e})",
            )
    return rows


def register_equality_reduction(checks: Checks) -> list[dict]:
    """I2: ``h = 0`` as ``h <= 0, -h <= 0`` recovers ``x*`` and ``y+ - y-``."""
    equality = qp_equality()
    reduced = qp_equality_reduced()
    m_eq = reduced.meta["m_eq"]
    y_eq = torch.as_tensor(reduced.meta["y_eq"], dtype=torch.get_default_dtype())

    rows = []
    for method in _methods():
        set_seed(0)
        x = reduced.make_params()[0]
        optimizer = method.build(reduced)
        primal = torch.optim.SGD(
            [x], lr=reduced.primal_step(method.penalty_coefficient))
        for _ in range(_iterations(reduced)):
            if not torch.isfinite(x).all():
                break
            optimizer.forward_update(reduced.objective([x]),
                                     reduced.constraints([x])).backward()
            primal.step()
            primal.zero_grad()

        duals = optimizer.duals.detach()
        difference = duals[:m_eq] - duals[m_eq:]
        x_error = float((x.detach()
                         - torch.as_tensor(reduced.x_star,
                                           dtype=x.dtype)).abs().max())
        y_error = float((difference - y_eq).abs().max())
        # A dual floor perturbs both multipliers of a pair, and the two floors
        # cancel in the difference, so it is not a reason to exclude a method.
        applies = _two_sided_reduction_applies(method)
        rows.append({"method": method.label,
                     "reduction documented for this method": applies,
                     "||x - x*||inf": x_error,
                     "||(y+ - y-) - y_eq||inf": y_error,
                     "y_eq": y_eq.numpy().round(4).tolist(),
                     "y+ - y-": difference.numpy().round(4).tolist()})
        if applies:
            checks.expect(
                y_error <= 1e-5 and x_error <= 1e-5,
                f"I2: {method.label} recovers the equality problem through the "
                f"h<=0,-h<=0 reduction",
                f"||(y+ - y-) - y_eq||inf {y_error:.3e}, "
                f"||x - x*||inf {x_error:.3e}",
            )
        else:
            checks.expect(
                True,
                f"I2: {method.label} is exempt — its documented route for an "
                f"equality is the threshold |h| <= tau, not the two-sided "
                f"reduction, and a penalty-barrier surrogate has no interior to "
                f"work in when both sides of a pair are active",
                f"reported, not gated: ||(y+ - y-) - y_eq||inf {y_error:.3e}, "
                f"||x - x*||inf {x_error:.3e}",
            )
    del equality
    return rows


def _two_sided_reduction_applies(method: Method) -> bool:
    """Whether ``h <= 0, -h <= 0`` is a documented route for this method."""
    return method.equality_via_two_sided


# --------------------------------------------------------------------------- #
# C — convergence, one untuned configuration per method
# --------------------------------------------------------------------------- #


def converge(problem: Problem, method: Method, iterations: int,
             record_at: set[int] | None = None) -> tuple[list[dict], dict]:
    set_seed(0)
    x = problem.make_params()[0]
    optimizer = method.build(problem)
    primal = torch.optim.SGD(
        [x], lr=problem.primal_step(method.penalty_coefficient))

    rows, final = [], {}
    diverged = False
    for k in range(iterations + 1):
        blown = k % 100 == 0 and not bool(
            torch.isfinite(x).all() and x.abs().max() < 1e12)
        last = k == iterations or blown
        if last or (record_at is not None and k in record_at):
            row = _metrics(problem, x, optimizer.duals.detach())
            row.update(iteration=k, problem=problem.name, method=method.label)
            row["relative KKT"] = _relative_kkt(problem, row)
            if not np.isfinite(row["stationarity"]) or row["violation"] > 1e8:
                diverged = True
            if record_at is not None:
                rows.append(row)
            if last or diverged:
                final = dict(row)
        if last or diverged:
            break

        primal.zero_grad()
        optimizer.forward_update(problem.objective([x]),
                                 problem.constraints([x])).backward()
        primal.step()

    final["diverged"] = diverged
    final["primal_lr"] = primal.param_groups[0]["lr"]
    final["dual_step"] = method.dual_step(problem)
    return rows, final


def register_convergence(checks: Checks, results: dict, methods: list[Method]) -> None:
    """Per-problem expectations, each conditioned on a structural predicate."""
    def residual(problem_name, method):
        return _relative_kkt_of(results[(problem_name, method.label)])

    def _relative_kkt_of(final):
        return float("inf") if final.get("diverged") else final["relative KKT"]

    # qp_active — unique KKT point, so every method must converge.
    for method in methods:
        value = residual("qp_active", method)
        checks.expect(
            value <= KKT_TOLERANCE,
            f"C/qp_active: {method.label} converges — the KKT point is unique "
            f"(strongly convex, all active, LICQ, strict complementarity), so "
            f"every method must",
            f"relative KKT {value:.3e}",
        )

    # qp_inactive — exact zeros are reachable iff the dual lower bound is 0.
    for method in methods:
        final = results[("qp_inactive", method.label)]
        bound = final.get("lower_bound") or 0.0
        value = _relative_kkt_of(final)
        if bound == 0.0:
            checks.expect(
                value <= KKT_TOLERANCE,
                f"C/qp_inactive: {method.label} has lower_bound=0, so it can "
                f"represent y*_i = 0 exactly and must converge",
                f"relative KKT {value:.3e}, smallest dual "
                f"{final['dual_min']:.3e}",
            )
        else:
            checks.expect(
                value >= 0.5 * bound,
                f"C/qp_inactive: {method.label} has lower_bound={bound:g} > 0, so "
                f"y*_i = 0 is unrepresentable and its error is bounded below by "
                f"that bound",
                f"relative KKT {value:.3e} against the bound {bound:g}",
            )

    # svm_iris — same structural rule as qp_inactive: an exact zero is reachable iff
    # the dual lower bound is 0. The singular bias direction turns out NOT to be an
    # obstruction to convergence, only to the *rate* -- which is why this problem
    # needs 213k iterations at its derived step rather than 13k.
    for method in methods:
        final = results[("svm_iris", method.label)]
        bound = final.get("lower_bound") or 0.0
        value = residual("svm_iris", method)
        if bound == 0.0:
            checks.expect(
                value <= KKT_TOLERANCE,
                f"C/svm_iris: {method.label} has lower_bound=0, so it can represent "
                f"the 96 zero multipliers exactly and must converge",
                f"relative KKT {value:.3e}",
                known_false=(
                    "the PI proportional term is what fails here, and this "
                    "experiment's original reasoning had it backwards. It predicted "
                    "that the bias direction's missing curvature would obstruct "
                    "plain dual ascent and that curvature *or* damping would rescue "
                    "it. Both halves are false: ALM(rho=0), with neither, reaches "
                    "9.1e-14, while nuPI(rho=0) at kp = ki stalls at 2.5e-01 -- "
                    "adding the proportional term without a quadratic term makes "
                    "this problem worse, not better. Note the failure is invisible "
                    "in stationarity and feasibility alone: with m=100 and n=5, "
                    "J' has a 95-dimensional null space, so only complementarity "
                    "pins y, and this configuration is feasible and stationary with "
                    "multipliers wrong by 0.25. Claimed only for kp = ki, the one "
                    "gain ratio tested."
                    if method.label == "nuPI (rho=0)" else None
                ),
            )
        else:
            checks.expect(
                value >= 0.5 * bound,
                f"C/svm_iris: {method.label} has lower_bound={bound:g} > 0, so the "
                f"96 zero multipliers are unrepresentable and its error is bounded "
                f"below by that bound",
                f"relative KKT {value:.3e} against the bound {bound:g}",
            )


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def make_figure(trajectories: dict, problems: list[Problem], methods: list[Method]):
    fig, axes, plt = figure(2, 2, row_height=2.1)
    for ax, problem in zip(axes, problems):
        for method in methods:
            rows = trajectories.get((problem.name, method.label))
            if not rows:
                continue
            finite = [r for r in rows if np.isfinite(r["relative KKT"])]
            ax.plot([r["iteration"] for r in finite],
                    [max(r["relative KKT"], 1e-17) for r in finite],
                    label=method.label, markevery=0.25, markersize=2.5,
                    **method.style)
        ax.axhline(KKT_TOLERANCE, color="0.6", lw=0.5, ls=(0, (1, 2)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{problem.name} (m={problem.m})")
        ax.set_xlabel("primal iteration")
        ax.set_ylabel("relative KKT residual")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_figure(fig, "e0a_convergence", EXPERIMENT)
    plt.close(fig)


def _record_at(iterations: int, points: int = 200) -> set[int]:
    grid = np.unique(np.geomspace(1, iterations, points).astype(int))
    return set(int(k) for k in grid) | {0, iterations}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="E0a: faithfulness of the dual optimizers")
    parser.add_argument("--iterations", type=int, default=None,
                        help="override the derived per-problem budget")
    parser.add_argument("--quick", action="store_true",
                        help="F, R and I only — the algebra, in seconds")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    use_float64()
    methods = _methods()
    checks = Checks(enabled=args.check)

    # ---- F ---------------------------------------------------------------- #
    print("F — fixed-point consistency at (x*, y*)")
    fixed_rows = [fixed_point(p, m) for p in _reference_problems() for m in methods]
    register_fixed_point(checks, fixed_rows)
    register_fixed_point_is_sharp(checks, qp_inactive())
    for row in fixed_rows:
        print(f"  {row['problem']:<14} {row['method']:<14} "
              f"grad_x={row['grad_x']:.2e}  excess drift={row['excess drift']:.2e}")
    write_table(fixed_rows, "e0a_fixed_point", EXPERIMENT,
                title="E0a/F: one forward_update from the exact KKT point. "
                      "'excess drift' is the multiplier movement a method is "
                      "answerable for, once its own declared lower_bound is allowed.")

    # ---- R ---------------------------------------------------------------- #
    print("\nR — exact reductions to ALM")
    reduction_rows = register_reductions(checks, _all_problems())
    for row in reduction_rows:
        print(f"  {row['problem']:<14} {row['reduction'][:34]:<36} "
              f"|dsurrogate|={row['surrogate difference']:.2e}  "
              f"|dy|max={row['max |dual difference|']:.2e}")
    write_table(reduction_rows, "e0a_reductions", EXPERIMENT,
                columns=["problem", "reduction", "surrogate difference",
                         "dual difference", "duals bitwise identical", "bar",
                         "max |surrogate difference|",
                         "max |dual difference|",
                         "steps with preconditions holding",
                         "precondition first broke at step", "note"],
                title="E0a/R: three exact reductions among four independently "
                      "written classes, one step and along a trajectory.")

    # ---- I ---------------------------------------------------------------- #
    print("\nI — invariances")
    scaling_rows = register_scaling_invariance(checks, methods)
    write_table(scaling_rows, "e0a_invariance_scaling", EXPERIMENT,
                title=f"E0a/I1: constraints scaled by {SCALE_ALPHA:g} must scale the "
                      f"multipliers by 1/{SCALE_ALPHA:g} and move nothing else.")
    equality_rows = register_equality_reduction(checks)
    for row in equality_rows:
        print(f"  equality reduction  {row['method']:<14} "
              f"|dy_eq|={row['||(y+ - y-) - y_eq||inf']:.2e}  "
              f"|dx|={row['||x - x*||inf']:.2e}")
    write_table(equality_rows, "e0a_invariance_equality", EXPERIMENT,
                columns=["method", "||(y+ - y-) - y_eq||inf", "||x - x*||inf",
                         "y_eq", "y+ - y-"],
                title="E0a/I2: an equality h=0 posed as h<=0, -h<=0. Individual "
                      "y+/y- are not determined, so only the difference is asserted.")

    if args.quick:
        main_exit(checks, EXPERIMENT, "e0a_predictions")
        return

    # ---- C ---------------------------------------------------------------- #
    print("\nC — convergence, one untuned configuration per method")
    problems = _all_problems()
    results, trajectories = {}, {}
    for problem in problems:
        for method in methods:
            budget = args.iterations or _iterations(problem)
            rows, final = converge(problem, method, budget,
                                   record_at=_record_at(budget))
            final["lower_bound"] = method.build(problem).param_groups[0].get(
                "lower_bound")
            results[(problem.name, method.label)] = final
            trajectories[(problem.name, method.label)] = rows
            print(f"  {problem.name:<14} {method.label:<14} "
                  f"relative KKT={final.get('relative KKT', float('nan')):.3e}  "
                  f"{_status(final)}  ({budget} it)")
    register_convergence(checks, results, methods)

    write_csv([r for rows in trajectories.values() for r in rows],
              "e0a_trajectories", EXPERIMENT)
    write_table(
        [{"problem": p.name, "method": m.label,
          "primal lr": results[(p.name, m.label)]["primal_lr"],
          "dual step": results[(p.name, m.label)]["dual_step"],
          "status": _status(results[(p.name, m.label)]),
          "relative KKT": results[(p.name, m.label)].get("relative KKT",
                                                         float("nan")),
          "||y-y*||inf": results[(p.name, m.label)].get("y_inf", float("nan")),
          "max [c]+": results[(p.name, m.label)]["violation"],
          "||grad f + J'y||inf": results[(p.name, m.label)]["stationarity"]}
         for p in problems for m in methods],
        "e0a_convergence", EXPERIMENT,
        title=f"E0a/C: one untuned configuration per method — primal step "
              f"1/(L_f + rho||J||^2), dual step 1/||J||^2, and a per-problem "
              f"iteration budget of {DUAL_STEP_BUDGET:g}/dual_step so that every "
              f"problem gets equal progress rather than equal iterations.",
    )

    # ---- O ---------------------------------------------------------------- #
    outcomes = []
    for problem in problems:
        statuses = [_status(results[(problem.name, m.label)]) for m in methods]
        outcomes.append({
            "problem": problem.name,
            "convex": problem.is_convex,
            "solved": statuses.count("solved"),
            "bounded (finite, not a KKT point)": statuses.count("bounded"),
            "diverged": statuses.count("diverged"),
            "did not solve": ", ".join(m.label for m, s in zip(methods, statuses)
                                       if s != "solved") or "-",
        })
    write_table(outcomes, "e0a_status", EXPERIMENT,
                title="E0a/O: outcome counts. On qp_nonconvex no convergence is "
                      "claimed — a fixed-penalty Lagrangian surrogate is unbounded "
                      "below on an indefinite objective, so failure there is a "
                      "limit of applicability, not a defect.")
    make_figure(trajectories, problems, methods)

    main_exit(checks, EXPERIMENT, "e0a_predictions")


if __name__ == "__main__":
    main()
