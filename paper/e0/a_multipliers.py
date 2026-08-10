"""
E0a — do the dual variables converge to the true Lagrange multipliers?

The claim under test is faithfulness, not performance: each dual optimizer in
``humancompatible.train.dual_optim`` should recover the multipliers of a problem
whose KKT point is known independently. Four problems, chosen so that the ways a
method can fail are separated:

* ``qp_active``    — strictly convex QP, every constraint active, every ``y*_i > 0``.
                     The easy case; failing here is a bug, not a limitation.
* ``qp_inactive``  — strictly convex QP where four of six multipliers are *exactly*
                     zero. Representing an exact zero is what the pre-``[c]_+``
                     quadratic penalty could not do.
* ``svm_iris``     — hard-margin SVM, 96 of 100 multipliers zero: the same test at
                     a realistic sparsity level, and with a *singular* objective
                     Hessian (no curvature in the bias).
* ``qp_nonconvex`` — indefinite QP over a box. No reference multipliers exist, so
                     only the KKT residual at the returned iterate is reported.

Protocol. One primal gradient step per dual update (the single-loop form all four
methods are published in; ``PBM`` gates its own dual step through
``primal_update_process_length``), driven through ``forward_update`` — the
package's documented training-loop entry point, and the ordering the published
recursions write, ``L_{t+1} = f_t + y_{t+1}'c_t``. The primal step is *derived*, not tuned:
``1/(L_f + rho ||J||^2)``, so every method gets the largest step that is safe for
its own surrogate, and a method carrying a large quadratic term visibly pays for
it. Each method's *dual* step is swept over a small grid and the value minimising
the final ``||y - y*||_inf`` is reported, so no result is an artifact of a step
size shared across methods. ``iALM`` is the exception: its ``beta`` is
simultaneously the quadratic coefficient and the dual step, so it is reported as
a labelled sweep rather than tuned away.

Predictions, registered before the first run (``--check`` fails the script if one
is violated):

P1 ``ALM(rho=1)`` and ``nuPI(rho=0)`` reach ``||y - y*||_inf <= 1e-4`` on all
   three reference problems.
   (The ``nuPI`` half of P1 is **false on svm_iris** and is left standing as
   registered. It was mis-scoped: the property that carries P1 is *having a
   quadratic term*, which ``nuPI`` by construction does not — so ``nuPI`` inherits
   exactly the bilinear-bias obstruction P2 identifies for ``ALM(rho=0)``, and its
   selected configuration turns out to be ``kp = 0``, i.e. plain dual gradient
   ascent. See P7.)
P2 ``ALM(rho=0)`` is plain gradient descent-ascent on the Lagrangian. On the two
   QPs the primal is strongly convex, which contracts, so it should also reach
   ``1e-4``. On ``svm_iris`` the Lagrangian is *bilinear* in the bias and its
   multiplier (the objective has no curvature in ``b``), where the last iterate
   of GDA need not converge — so ``ALM(rho=0)`` is predicted to do strictly worse
   there than ``ALM(rho=1)``.
P3 ``PBM`` cannot represent ``y_i = 0``: its dual update is multiplicative and its
   ``dual_range`` floor is ``1e-4``. Its ``||y - y*||_inf`` should therefore floor
   at roughly that value on the two problems with zero multipliers, while still
   converging on ``qp_active`` where every multiplier is positive. A predicted
   consequence of strict positivity, not a defect.
P4 ``iALM`` cannot drop its quadratic term, so its best ``beta`` should not match
   ``ALM(rho=1)`` on ``svm_iris``, and its final error should vary systematically
   with ``beta``.
P5a Every method ends feasible (``max [c]_+ <= 1e-6``) on the three convex problems.
   **False**, and recorded as such rather than edited: it is subsumed by P1, P2 and
   P6, since a configuration that has not converged is not feasible either.
P5b The sharper, non-redundant half of P5a: on every convex problem, every method
   that *did* recover ``y*`` to ``1e-4`` is also feasible to ``1e-8``, i.e. the two
   halves of the KKT conditions arrive together. Registered alongside P5a rather
   than in place of it.
P6 On ``qp_nonconvex``, exactly the configurations whose quadratic coefficient
   exceeds ``-lambda_min(Q) = 2.2`` stay bounded. Added after a smoke run showed
   every configuration diverging there, but derived rather than fitted: over the
   box the surrogate's Hessian is ``Q + rho I`` on the violated faces, so it is
   bounded below only once ``rho > 2.2``, which ``rho = 1`` provably cannot
   satisfy. ``ALM(rho=10)`` was added to the method list for this reason.
P7 A cross-implementation consistency check, not a claim about a method:
   ``nuPI(kp=0)`` is algebraically ``y <- y + ki*c``, which is ``ALM(lr=ki,
   rho=0)``. The two independent implementations must therefore produce **bitwise
   identical** iterates. This is what explains P1's ``nuPI`` failure — the swept
   configuration that wins on ``svm_iris`` is exactly ``kp = 0`` — and it is a
   genuine test of both dual-update code paths against each other.
P8 ``nuPI(rho=1)`` reaches ``1e-4`` on all three reference problems. Added after
   P1's ``nuPI`` half failed, and it is the *discriminating* test between the two
   candidate explanations of that failure: either the PI dual rule is inadequate
   on ``svm_iris``, or the obstruction is the objective's missing curvature in the
   bias and any quadratic term supplies it. P2 argues for the second; if
   ``nuPI(rho=1)`` converges where ``nuPI(rho=0)`` does not, that settles it, and
   ``rho > 0`` becomes a documented recommendation rather than a deviation from the
   reference method. (The ``nuPI`` paper defines a multiplier update, not an
   augmented surrogate, so ``rho=0`` remains the published form and stays in the
   comparison.)

A note on the reference values. ``svm_iris``'s ``y*`` is *not* taken from the QP
solver's output, which is accurate to only ~1e-9 and would become a floor that
several methods tie at for reasons that have nothing to do with the methods. The
solver is used only to identify the active set; the multipliers then come from an
exact solve of the resulting equality-constrained KKT system. The two QPs are
constructed backwards from a chosen ``(x*, y*)``, so their references are exact by
construction.

Usage::

    python paper/e0/a_multipliers.py            # full run
    python paper/e0/a_multipliers.py --quick    # seconds, for smoke testing
    python paper/e0/a_multipliers.py --check    # exit non-zero on a failed prediction
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
from paper.problems.qp import qp_active, qp_inactive, qp_nonconvex
from paper.problems.svm import svm_iris

EXPERIMENT = "e0a"

# Dual step sizes swept for the methods whose dual step is a free parameter.
# The grid reaches 1e-4 because a method with no quadratic term is driven to the
# smallest step it is offered on svm_iris, and a grid that ends at its own choice
# is a truncated grid, not a tuned method.
DUAL_STEPS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
# PBM's "dual step" is the smoothing factor gamma of y <- gamma*y + (1-gamma)*y*phi'(c/p):
# small gamma means a fast, nearly multiplicative update.
PBM_GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9]
# nuPI's proportional gain, as a multiple of its integral gain. 0 reduces the
# controller to plain dual gradient ascent, which is the sanity check that the PI
# terms are what makes the difference.
NUPI_GAIN_RATIOS = [0.0, 1.0, 10.0]


# --------------------------------------------------------------------------- #
# method configurations
# --------------------------------------------------------------------------- #


@dataclass
class Method:
    """One line in the figure: a constructor plus the grid of dual settings.

    :param build: Called as ``build(problem, config)`` where ``config`` is one
        entry of ``sweep``, spelled as the constructor's own keyword names so the
        results table reports what was actually passed.
    :param penalty_coefficient: The quadratic coefficient this configuration
        carries, used *only* to derive the primal step size. Reporting it is the
        point: it is the price of the quadratic term.
    """

    label: str
    build: Callable[[Problem, dict], DualOptimizer]
    penalty_coefficient: float
    sweep: list[dict]
    style: dict = field(default_factory=dict)


def _config_label(config: dict) -> str:
    return ", ".join(f"{key}={value:g}" for key, value in config.items())


def _methods() -> list[Method]:
    alm_shades = {0.0: ("#7FB2D0", "--", "o"), 1.0: ("#2E86AB", "-", "s"),
                  10.0: ("#1B4965", "-.", "*")}
    methods = [
        Method(
            # rho=10 is here for qp_nonconvex: a Lagrangian method keeps the
            # iterates bounded on an indefinite problem only once its quadratic
            # term dominates the negative curvature, and lambda_min(Q) = -2.2
            # there, so rho = 1 provably cannot. Run on every problem for
            # consistency.
            f"ALM (rho={rho:g})",
            lambda p, cfg, rho=rho: ALM(
                m=p.m, penalty=rho, init_duals=0.0, is_ineq=True, **cfg
            ),
            penalty_coefficient=rho,
            sweep=[{"lr": lr} for lr in DUAL_STEPS],
            style={"color": color, "ls": ls, "marker": marker},
        )
        for rho, (color, ls, marker) in alm_shades.items()
    ]
    # rho=0 is the published method (the nuPI paper defines a multiplier update,
    # not an augmented surrogate, and 0 is the shipped default). rho=1 is here to
    # separate two candidate explanations of P1's failure: is the obstruction on
    # svm_iris the *dual rule*, or the missing curvature in the bias? If a
    # quadratic term fixes nuPI exactly as it fixes ALM, it is the latter. See P8.
    methods += [
        Method(
            f"nuPI (rho={rho:g})",
            lambda p, cfg, rho=rho: nuPI(
                m=p.m, nu=0.01, penalty=rho, init_duals=0.0, is_ineq=True, **cfg
            ),
            penalty_coefficient=rho,
            sweep=[{"ki": ki, "kp": ratio * ki}
                   for ki in DUAL_STEPS for ratio in NUPI_GAIN_RATIOS],
            style={"color": color, "ls": ls, "marker": "v"},
        )
        for rho, color, ls in [(0.0, "#E0A458", "--"), (1.0, "#A8621B", "-")]
    ]
    methods.append(
        Method(
            "PBM",
            lambda p, cfg: PBM(
                m=p.m,
                penalty_mult=0.1,
                delta=1.0,
                penalty_update="dimin_adapt",
                gamma_annealing=False,
                penalty_annealing=False,
                **cfg,
            ),
            # phi'' <= 1 for the quadratic-logarithmic barrier, so the surrogate's
            # curvature is bounded by max_i (y_i / p_i) ||J||^2; 1.0 is the value
            # that holds at the solutions here (y* = O(1), p in [0.1, 1]).
            penalty_coefficient=1.0,
            sweep=[{"gamma": gamma} for gamma in PBM_GAMMAS],
            style={"color": "#D1495B", "ls": "-", "marker": "D"},
        )
    )
    # beta is simultaneously the quadratic coefficient and the dual step, so it
    # cannot be tuned away: each value gets its own line.
    methods += [
        Method(
            f"iALM (beta={beta:g})",
            lambda p, cfg: iALM(
                m=p.m, sigma=1.0, gamma=1.0, init_duals=0.0, is_ineq=True, **cfg
            ),
            penalty_coefficient=beta,
            sweep=[{"beta": beta}],
            style={"color": shade, "ls": ":", "marker": "^"},
        )
        for beta, shade in zip([0.1, 1.0, 10.0], ["#9BD1C4", "#5BC0BE", "#347B7A"])
    ]
    return methods


def _problems() -> list[Problem]:
    return [qp_active(), qp_inactive(), svm_iris(), qp_nonconvex()]


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def _metrics(problem: Problem, x: torch.Tensor, duals: torch.Tensor) -> dict:
    """KKT diagnostics at ``(x, y)``, computed exactly (no minibatch noise)."""
    f = problem.objective([x])
    c = problem.constraints([x])
    # grad_x (f + y'c) = grad f + J'y, in one backward pass and without ever
    # forming the m-by-n Jacobian.
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
        row["y_rel"] = float(error.norm() / y_star.norm())
    return row


def run(
    problem: Problem,
    method: Method,
    config: dict,
    iterations: int,
    record_at: set[int] | None = None,
) -> tuple[list[dict], dict]:
    """Run one (problem, method, dual-configuration) triple.

    :param record_at: Iterations at which to store a trajectory row. ``None``
        records only the last iterate, which is all the sweep needs.
    :return: ``(rows, final)``.
    """
    set_seed(0)
    x = problem.make_params()[0]
    dual = method.build(problem, config)
    primal = torch.optim.SGD([x], lr=problem.primal_step(method.penalty_coefficient))

    rows, final = [], {}
    diverged = False
    for k in range(iterations + 1):
        # Cheap blow-up check: stop as soon as the iterate is beyond saving rather
        # than spending the remaining iterations overflowing to inf, which also
        # keeps the reported violation a finite (huge) number instead of nan.
        blown = k % 100 == 0 and not bool(
            torch.isfinite(x).all() and x.abs().max() < 1e12
        )
        last = k == iterations or blown
        if last or (record_at is not None and k in record_at):
            row = _metrics(problem, x, dual.duals.detach())
            row.update(iteration=k, problem=problem.name, method=method.label,
                       config=_config_label(config))
            if not np.isfinite(row["stationarity"]) or row["violation"] > 1e8:
                diverged = True
            if record_at is not None:
                rows.append(row)
            if last or diverged:
                final = dict(row)
        if last or diverged:
            break

        loss = problem.objective([x])
        constraints = problem.constraints([x])
        primal.zero_grad()
        # forward_update is the package's documented training-loop entry point, and
        # it is also what the published recursions write: the multipliers advance
        # first and the surrogate is formed with the *new* ones,
        # L_{t+1} = f_t + y_{t+1}'c_t. (PBM is unaffected either way — it overrides
        # _snapshot to take a pre-update copy, so both orderings coincide for it.)
        #
        # Using forward() and update() separately is also supported, with one
        # constraint: .backward() must come before update(). forward() builds the
        # surrogate from the live dual tensor, autograd keeps that tensor for the
        # y'c backward, and update() mutates it in place. The primal step may go on
        # either side of update() -- the constraint tensor already holds the values
        # from this iterate, so that choice is bitwise immaterial.
        dual.forward_update(loss, constraints).backward()
        primal.step()

    final["diverged"] = diverged
    final["primal_lr"] = primal.param_groups[0]["lr"]
    return rows, final


def _record_at(iterations: int, points: int = 200) -> set[int]:
    """Log-spaced recording points, so a 20k-iteration trajectory stays a small CSV."""
    grid = np.unique(np.geomspace(1, iterations, points).astype(int))
    return set(int(k) for k in grid) | {0, iterations}


def _score(final: dict) -> float:
    """Sweep objective: final multiplier error, or the KKT residual without a reference."""
    if final.get("diverged"):
        return float("inf")
    if "y_inf" in final:
        return final["y_inf"]
    return max(final["stationarity"], final["violation"])


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def make_figure(trajectories: dict, problems: list[Problem], methods: list[Method]):
    fig, axes, plt = figure(2, 2, row_height=2.1)
    for ax, problem in zip(axes, problems):
        reference = problem.has_reference_multipliers
        key = "y_inf" if reference else "stationarity"
        for method in methods:
            rows = trajectories.get((problem.name, method.label))
            if not rows:
                continue
            # Truncate at the last finite value: a diverged run's tail is inf/nan
            # and would silently drop the whole line rather than showing where it
            # left the plot.
            finite = [r for r in rows if np.isfinite(r[key])]
            iterations = [r["iteration"] for r in finite]
            # 1e-16 floor so an exactly-zero error is still drawn on a log axis.
            values = [max(r[key], 1e-16) for r in finite]
            ax.plot(iterations, values, label=method.label, markevery=0.25,
                    markersize=2.5, **method.style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{problem.name} (m={problem.m})")
        ax.set_xlabel("primal iteration")
        ax.set_ylabel(r"$\|y_k - y^\star\|_\infty$" if reference
                      else r"$\|\nabla f + J^\top y\|_\infty$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save_figure(fig, "e0a_multipliers", EXPERIMENT)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #

TOLERANCE = 1e-4


def register_predictions(checks: Checks, best: dict, problems: list[Problem]) -> None:
    """Evaluate P1-P5 against the selected configurations."""
    reference_problems = [p.name for p in problems if p.has_reference_multipliers]
    convex_problems = [p.name for p in problems if p.is_convex]

    def error(method, problem):
        return _score(best[(problem, method)])

    # P1
    for label in ("ALM (rho=1)", "nuPI (rho=0)"):
        for problem in reference_problems:
            value = error(label, problem)
            checks.expect(
                value <= TOLERANCE,
                f"P1: {label} reaches ||y-y*||inf <= {TOLERANCE:g} on {problem}",
                f"got {value:.3e}",
                known_false=(
                    "P1 was mis-scoped for nuPI: it has no quadratic term, so it "
                    "inherits the bilinear-bias obstruction of P2 rather than "
                    "escaping it. Its selected configuration here is kp=0, i.e. "
                    "plain dual gradient ascent (see P7). P8 tests the implied "
                    "remedy."
                    if label == "nuPI (rho=0)" and problem == "svm_iris" else None
                ),
            )

    # P2
    for problem in ("qp_active", "qp_inactive"):
        value = error("ALM (rho=0)", problem)
        checks.expect(
            value <= TOLERANCE,
            f"P2: ALM (rho=0) reaches ||y-y*||inf <= {TOLERANCE:g} on {problem} "
            f"(strongly convex primal)",
            f"got {value:.3e}",
        )
    plain, augmented = error("ALM (rho=0)", "svm_iris"), error("ALM (rho=1)", "svm_iris")
    checks.expect(
        plain > augmented,
        "P2: on svm_iris, whose Lagrangian is bilinear in the bias, ALM (rho=0) "
        "does worse than ALM (rho=1)",
        f"rho=0 {plain:.3e} vs rho=1 {augmented:.3e}",
    )

    # P3
    checks.expect(
        error("PBM", "qp_active") <= TOLERANCE,
        f"P3: PBM reaches {TOLERANCE:g} on qp_active, where every y*_i > 0",
        f"got {error('PBM', 'qp_active'):.3e}",
    )
    for problem in ("qp_inactive", "svm_iris"):
        value = error("PBM", problem)
        floor = best[(problem, "PBM")]["dual_min"]
        checks.expect(
            value >= 0.5e-4,
            f"P3: PBM floors near its dual_range lower bound on {problem}",
            f"||y-y*||inf {value:.3e}, smallest dual {floor:.3e}",
        )

    # P4
    ialm_errors = [
        (beta, error(f"iALM (beta={beta:g})", "svm_iris")) for beta in (0.1, 1.0, 10.0)
    ]
    best_ialm = min(value for _, value in ialm_errors)
    checks.expect(
        best_ialm > augmented,
        "P4: iALM cannot drop its quadratic term, so its best beta does not match "
        "ALM (rho=1) on svm_iris",
        "; ".join(f"beta={b:g}: {v:.3e}" for b, v in ialm_errors)
        + f"; ALM(rho=1) {augmented:.3e}",
    )
    ordered = [value for _, value in ialm_errors]
    monotone = ordered == sorted(ordered) or ordered == sorted(ordered, reverse=True)
    checks.expect(
        monotone,
        "P4: iALM's final multiplier error varies monotonically with beta on svm_iris",
        "; ".join(f"beta={b:g}: {v:.3e}" for b, v in ialm_errors),
    )

    # P5a — the blanket form, as first registered. One check over all convex
    # problems, so the single recorded explanation covers the single claim.
    offenders = {
        (name, method): best[(name, method)]["violation"]
        for (name, method) in best
        if name in convex_problems and best[(name, method)]["violation"] > 1e-6
    }
    checks.expect(
        not offenders,
        "P5a: every method ends feasible (max [c]+ <= 1e-6) on every convex problem",
        "; ".join(f"{n}/{m}: {v:.2e}" for (n, m), v in offenders.items()),
        known_false=(
            "the blanket form is subsumed by P1/P2/P6: a configuration that has "
            "not converged is not feasible either, so this only restates their "
            "failures. The claim with content is P5b."
        ),
    )

    # P5b — the same claim restricted to the methods that actually converged,
    # which is the part not already entailed by P1 and P2. Kept alongside P5a
    # rather than replacing it, so a failure of the blanket form stays on record.
    for problem in convex_problems:
        offenders = {
            method: best[(problem, method)]["violation"]
            for (name, method) in best
            if name == problem
            and _score(best[(problem, method)]) <= TOLERANCE
            and best[(problem, method)]["violation"] > 1e-8
        }
        checks.expect(
            not offenders,
            f"P5b: on {problem}, every method that recovered y* is also feasible "
            f"to 1e-8 (feasibility and multiplier accuracy arrive together)",
            "; ".join(f"{k}: {v:.2e}" for k, v in offenders.items()),
        )

    # P6 — added after a smoke run showed every method diverging on qp_nonconvex,
    # and derived rather than fitted: a Lagrangian surrogate over the box has
    # Hessian Q + rho*I on the violated faces, so it is bounded below only once
    # rho exceeds -lambda_min(Q) = 2.2. rho = 1 provably cannot; rho = 10 and
    # beta = 10 can.
    survivors = {
        method for (name, method) in best
        if name == "qp_nonconvex" and not best[(name, method)]["diverged"]
    }
    expected = {"ALM (rho=10)", "iALM (beta=10)"}
    checks.expect(
        survivors == expected,
        "P6: on qp_nonconvex exactly the configurations whose quadratic "
        "coefficient exceeds -lambda_min(Q) = 2.2 stay bounded",
        f"bounded: {sorted(survivors)}; expected: {sorted(expected)}",
    )

    # P7
    check_nupi_reduces_to_alm(checks, problems)

    # P8 — the remedy implied by P1's failure and P2's explanation.
    for problem in reference_problems:
        value = error("nuPI (rho=1)", problem)
        checks.expect(
            value <= TOLERANCE,
            f"P8: nuPI(rho=1) reaches ||y-y*||inf <= {TOLERANCE:g} on {problem}, so "
            f"the svm_iris obstruction is the missing curvature in the bias and not "
            f"the PI dual rule",
            f"got {value:.3e}"
            + (f" (nuPI(rho=0): {error('nuPI (rho=0)', problem):.3e}, "
               f"ALM(rho=1): {error('ALM (rho=1)', problem):.3e})"
               if problem == "svm_iris" else ""),
        )


def check_nupi_reduces_to_alm(checks: Checks, problems: list[Problem],
                              iterations: int = 500, step: float = 0.01) -> None:
    """``nuPI(kp=0)`` and ``ALM(rho=0, lr=ki)`` are the same recursion.

    Run at a matched, fixed setting rather than at whatever the sweep happened to
    select, so the check means the same thing regardless of the sweep's outcome.
    """
    alm = Method("ALM (rho=0)",
                 lambda p, cfg: ALM(m=p.m, penalty=0.0, init_duals=0.0,
                                    is_ineq=True, **cfg),
                 penalty_coefficient=0.0, sweep=[{"lr": step}])
    pi = Method("nuPI (kp=0)",
                lambda p, cfg: nuPI(m=p.m, nu=0.01, penalty=0.0, init_duals=0.0,
                                    is_ineq=True, **cfg),
                penalty_coefficient=0.0, sweep=[{"ki": step, "kp": 0.0}])
    for problem in problems:
        _, a = run(problem, alm, alm.sweep[0], iterations)
        _, b = run(problem, pi, pi.sweep[0], iterations)
        keys = ("f", "violation", "stationarity", "dual_min")
        identical = all(a[key] == b[key] for key in keys)
        checks.expect(
            identical,
            f"P7: nuPI(kp=0) is bitwise identical to ALM(rho=0, lr={step:g}) on "
            f"{problem.name} — the same recursion through two implementations",
            "; ".join(f"{k}: {a[k]!r} vs {b[k]!r}" for k in keys if a[k] != b[k]),
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--quick", action="store_true",
                        help="200 iterations and a two-point dual sweep")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if a registered prediction fails")
    parser.add_argument("--problems", nargs="*", default=None)
    args = parser.parse_args(argv)

    use_float64()
    iterations = 200 if args.quick else args.iterations

    problems = _problems()
    if args.problems:
        problems = [p for p in problems if p.name in args.problems]
    methods = _methods()
    if args.quick:
        for method in methods:
            if len(method.sweep) > 1:
                method.sweep = [method.sweep[0], method.sweep[len(method.sweep) // 2]]

    sweep_rows, best, trajectories = [], {}, {}
    for problem in problems:
        print(f"\n{problem.name}: m={problem.m}, {problem.notes}")
        for method in methods:
            scored = []
            for config in method.sweep:
                _, final = run(problem, method, config, iterations)
                sweep_rows.append(final)
                scored.append((_score(final), config, final))
            score, config, final = min(scored, key=lambda item: item[0])
            best[(problem.name, method.label)] = final
            print(f"  {method.label:<16} {_config_label(config):<22} "
                  f"score={score:.3e}  lr={final['primal_lr']:.2e}"
                  + ("  DIVERGED" if final["diverged"] else ""))
            # Re-run the winner with recording on. Same seed, same code path, so
            # the trajectory's last row must reproduce the sweep's final row.
            rows, replay = run(problem, method, config, iterations,
                               record_at=_record_at(iterations))
            assert _score(replay) == score, "replay of the selected run diverged"
            trajectories[(problem.name, method.label)] = rows

    write_csv([row for rows in trajectories.values() for row in rows],
              "e0a_trajectories", EXPERIMENT)
    write_csv(sweep_rows, "e0a_sweep", EXPERIMENT)

    table = []
    for problem in problems:
        for method in methods:
            final = best[(problem.name, method.label)]
            table.append({
                "problem": problem.name,
                "method": method.label,
                "dual config": final["config"],
                "primal lr": final["primal_lr"],
                "status": "diverged" if final["diverged"] else "ok",
                "||y-y*||inf": final.get("y_inf", float("nan")),
                "||y-y*||2/||y*||2": final.get("y_rel", float("nan")),
                "max [c]+": final["violation"],
                "||grad f + J'y||inf": final["stationarity"],
                "f - f*": final.get("f_gap", float("nan")),
            })
    write_table(
        table, "e0a_final", EXPERIMENT,
        columns=["problem", "method", "dual config", "primal lr", "status",
                 "||y-y*||inf", "||y-y*||2/||y*||2", "max [c]+",
                 "||grad f + J'y||inf", "f - f*"],
        title=f"E0a: multiplier recovery after {iterations} primal iterations",
    )
    make_figure(trajectories, problems, methods)

    checks = Checks(enabled=args.check)
    if not args.problems and not args.quick:
        register_predictions(checks, best, problems)
    main_exit(checks)


if __name__ == "__main__":
    main()
