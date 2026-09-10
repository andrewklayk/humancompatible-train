"""
E0e -- cross-validation of ``ALM(penalty=0)`` against Cooper.

E0a validates our dual optimizers against exact reference solutions and against
*each other*. Neither answers the question an MPC reviewer will ask about a
software paper: how does this compare to `Cooper <https://github.com/cooper-org/cooper>`_
(``cooper-optim``), the PyTorch constrained-optimization library from the same
group as arXiv:2208.04425 and the one package with the same remit as
``dual_optim``? §1.3 of the paper claims ``ALM(penalty=0, is_ineq=True)`` *is*
that paper's projected GDA. Until now that claim rested on reading our own code.

So: pin both libraries to the same algorithm and compare trajectories, then
anchor both against the exact KKT point -- "agree *and* are right", not merely
"agree with each other". Scope is deliberately one method,
``ALM(penalty=0)`` <-> Cooper's ``Lagrangian`` + SGD-ascent multiplier.

Which Cooper optimizer is the right counterpart is not obvious, and getting it
wrong would silently compare two different algorithms. Our ``_snapshot`` returns
the *live* dual tensor, and ``_dual_update`` + the safeguard clamp run *before*
``_add_contributions``, so::

    forward_update:            y_{t+1} = [y_t + g*c_t]_+ ;  L = f(x_t) + y_{t+1}'c_t
    forward -> backward -> update:                          L = f(x_t) + y_t'c_t

Cooper's ``SimultaneousOptimizer`` fills both ``.grad`` buffers from one forward
at ``(x_t, y_t)``, so its primal gradient uses **pre-update** multipliers; its
``AlternatingDualPrimalOptimizer`` steps the dual *before* building the primal
Lagrangian, from the same single forward, so its primal gradient uses
**post-update** multipliers; and its ``AlternatingPrimalDualOptimizer`` -- the one
Cooper's own documentation recommends -- steps the primal first and then
**re-evaluates the constraints at** ``x_{t+1}`` to drive the dual, the only one of
the three that costs a second constraint evaluation. Hence the pairing

=============================================  ====================================
ours                                           Cooper
=============================================  ====================================
``forward_update(...).backward()``             ``AlternatingDualPrimalOptimizer``
``forward(...)`` / ``.backward()`` /           ``SimultaneousOptimizer``
``primal.step()`` / ``update(c)``
``forward(...)`` / ``.backward()`` /           ``AlternatingPrimalDualOptimizer``
``primal.step()`` / ``update(c(x_{t+1}))``
=============================================  ====================================

and every mismatched pairing is a negative control for the matched ones.

The third row is the one this script had to be extended for: ``dual_optim`` has no
primal-dual *entry point*, but it does not need one -- the ordering is the split
API with the constraint closure called a second time after ``primal.step()``,
which is the loop a user writes anyway. Whether that ordering is *worth* its
second constraint evaluation is a separate question, and prediction X8 answers it
as an identity rather than a horse race.

The dual updates are the *same floating-point operation* on both sides -- ours is
``duals.add_(c, alpha=lr)`` then ``clamp_``; Cooper's dual scalar is
``einsum("i...,i...->", y, c.detach())`` (a **sum**, no ``1/m``), giving
``y.grad = c`` exactly, stepped by ``SGD(maximize=True)`` into
``add_(grad, alpha=+lr)`` and then projected by ``post_step_``'s ``relu``. So
bitwise is the right bar for the duals. The primal side is held to rounding
instead: ``snapshot @ c`` and Cooper's ``einsum`` are different kernels.

Predictions
-----------

**X -- parity** (200 steps, five problems, from a common start point):

X1  ``forward_update`` == ``AlternatingDualPrimalOptimizer``: duals bitwise,
    parameters within ``ALGEBRAIC``.
X2  ``forward``/``backward``/``update`` == ``SimultaneousOptimizer``, same bars.
X3  *negative control*: each ``ours`` side, paired with the *wrong* Cooper class,
    must diverge by far more than ``ALGEBRAIC``. Without this, X1, X2 and X7
    could all pass merely because every configuration converges to the same point.
X5  ``relu`` and ``clamp_`` agree at exactly zero: started from ``y_0 = 0.5`` on
    ``qp_inactive``, whose ``y*`` has exact zeros, both libraries pin the *same*
    multipliers to exactly ``0.0``.
X7  ``forward``/``backward``/``primal.step()``/``update(c(x_{t+1}))`` ==
    ``AlternatingPrimalDualOptimizer``, same bars. Cooper's recommended optimizer
    is reachable from ``dual_optim`` without a new entry point.

**P -- is the recommended ordering a different algorithm?** (parity horizon):

X8  It is not. Both orderings are the *same* recursion ``y <- P(y + g*c(x_t))``
    over the same constraint sequence; dual-primal simply folds ``c(x_0)`` in one
    step earlier. So ``AlternatingPrimalDualOptimizer`` started from
    ``y_0' = [g*c(x_0)]_+`` must reproduce ``forward_update`` started from
    ``y_0 = 0`` exactly: parameters within ``ALGEBRAIC``, duals bitwise *at an
    offset of one step*. Reported alongside is the same-index dual gap, which has
    to exceed ``CONTROL_FLOOR`` or the offset would be doing no work and the
    claim would be vacuous. The practical reading -- one dual step of lag, paid
    for with twice the constraint evaluations -- is X9.

**K -- KKT anchor** (long run, skipped under ``--quick``):

X4  Run to convergence or to a cap: both libraries reach the same relative KKT
    residual *and* clear ``KKT_TOLERANCE`` at the same iteration -- the stopping
    test is part of the comparison, so no fixed budget has to be defended. Then,
    once and without naming a problem, at least one problem is driven to the
    exact ``(x*, y*)``, which is what stops every agreement above from being
    agreement in being wrong together. Which problems converge under a single
    fixed step size is E0a's question (its finding #4: with one dual step the
    iterations needed scale like ``||J||^2``, so a shared budget is not equal
    progress) and deliberately not re-answered here.
X9  From a common ``y_0 = 0``, the primal-dual ordering is never *ahead* of
    dual-primal: it needs at least as many iterations to clear the tolerance,
    or -- where neither clears it -- ends at no lower a residual, while spending
    two constraint evaluations per step against one. This runs on our side only,
    which X1/X2/X7 licenses: each ordering is bitwise its Cooper counterpart, so
    running one library runs both.

**S -- stochastic constraint** (``income_pairwise``, skipped under ``--quick``
and when the ACS data is absent):

X6  With identical batches and identical initial weights, a minibatch-stochastic
    data-dependent constraint (m = 30) gives the same trajectory in both
    libraries. Batches are materialised once and replayed into both runs, and
    one ``state_dict`` is loaded into both models -- by construction, not by
    trusting two shufflers to agree. That is E2a's shared-generator bug, which
    cost a day, encoded as a method.

**O -- observations, reported not gated:** per-step cost of each library's dual
layer, and an API-surface comparison.

Two facts about Cooper 1.0.1 that the paper should record either way, both
established by reading its source rather than inferred from these numbers:

* **It has no dual restart.** ``grep -i restart`` hits only the *penalty
  coefficient* updaters; nothing zeroes ``y``. Cooper 0.x had ``dual_restarts``.
  So ``ALM(restart=True)`` -- arXiv:2208.04425 Eq. (6) -- is not available in the
  authors' current library, which is where E2a's restart result gets to stand.
* **It ships its own** ``nuPI`` (``cooper.optim.nuPI``), usable directly as a
  dual optimizer. A reference implementation for our ``nuPI`` row, and the
  natural next step past this script -- deliberately out of scope here.

Usage::

    python paper/e0/e_cooper.py --quick --check   # parity only, seconds
    python paper/e0/e_cooper.py --check           # + KKT anchor, fairness, overhead
    python paper/e0/e_cooper.py                   # full run, exit 0 regardless
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import math
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from humancompatible.train.dual_optim import ALM
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
)
from paper.problems.svm import svm_iris

try:
    import cooper

    HAVE_COOPER = True
except ImportError:  # pragma: no cover - optional [compare] extra
    HAVE_COOPER = False

EXPERIMENT = "e0e"

# Same tolerance policy as E0a: these are identities, so the only admissible
# error is floating-point rounding.
ALGEBRAIC = 64 * float(np.finfo(np.float64).eps)
KKT_TOLERANCE = 1e-6
# X3 asserts the *mismatched* pairing is distinguishable. One dual step apart is
# a difference of order lr*|c|, so anything near rounding would mean the two
# orderings are not actually different and X1/X2 prove nothing.
CONTROL_FLOOR = 1e-6

# The three step orderings, named once so a table cell and a branch cannot drift.
DUAL_PRIMAL = "dual_primal"      # ours: forward_update
SIMULTANEOUS = "simultaneous"    # ours: split forward / backward / update(c)
PRIMAL_DUAL = "primal_dual"      # ours: split, with c re-evaluated at x_{t+1}
# Constraint evaluations each ordering costs per step -- the price of primal-dual.
EVALUATIONS = {DUAL_PRIMAL: 1, SIMULTANEOUS: 1, PRIMAL_DUAL: 2}

PRIMAL_LR = 5e-3          # E0a's constant, so the two scripts are comparable
ALM_DUAL_LR = 0.01        # ALM's shipped default
PARITY_STEPS = 200
ANCHOR_CAP = 20_000       # hard cap; not reaching the tolerance is reported
CHECK_EVERY = 100         # how often the stopping test is evaluated, as in E0a
OVERHEAD_STEPS = 2_000

# E2a's configuration for the fairness problem, so X6 measures the same pipeline.
FAIR_PRIMAL_LR = 1e-3
FAIR_DUAL_LR = 0.05
FAIR_STEPS = 200
FAIR_BOUND = 0.05


# --------------------------------------------------------------------------- #
# the two libraries, as a user writes them
# --------------------------------------------------------------------------- #
#
# Deliberately four small functions, two per library, doing exactly the same job:
# they are what O2 counts, so any asymmetry in the count has to be real rather
# than an artefact of how the comparison was written.


def _setup_ours(problem: Problem, *, init: float = 0.0, equality: bool = False):
    """``ALM(penalty=0)`` on ``problem``: our library's whole setup."""
    x = problem.make_params()[0]
    dual = ALM(
        m=problem.m, lr=ALM_DUAL_LR, penalty=0.0, init_duals=init,
        is_ineq=not equality,
        # ALM ships a safeguard box, dual_range=(-100, 100), that Cooper has no
        # equivalent of: its projection is relu, i.e. [0, inf). Opening the box
        # is what makes the two libraries the same algorithm -- and the box is a
        # real design difference, so it is opened explicitly here rather than
        # left to not bind by luck.
        dual_range=(-math.inf, math.inf) if equality else (0.0, math.inf),
        momentum=0.0, dampening=0.0, restart=False,
    )
    primal = torch.optim.SGD([x], lr=PRIMAL_LR)
    return x, dual, primal


def _step_ours(problem: Problem, state, *, ordering: str = DUAL_PRIMAL) -> None:
    """One step in one of the three orderings the two libraries share."""
    x, dual, primal = state
    loss, c = problem.objective([x]), problem.constraints([x])
    primal.zero_grad()
    if ordering == DUAL_PRIMAL:
        dual.forward_update(loss, c).backward()
        primal.step()
    else:
        dual.forward(loss, c).backward()
        primal.step()
        if ordering == PRIMAL_DUAL:
            # The dual is to follow the *new* iterate, so the constraints are
            # re-evaluated there. No graph is needed -- the dual update reads
            # values only -- which is what makes this one closure call rather
            # than a second full forward.
            with torch.no_grad():
                c = problem.constraints([x])
        dual.update(c)


def _setup_cooper(problem: Problem, *, init: float = 0.0, equality: bool = False):
    """The same thing in Cooper: a CMP, a constraint, a multiplier, an optimizer."""
    x = problem.make_params()[0]

    class Wrapped(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()
            self.constraint = cooper.Constraint(
                constraint_type=(cooper.ConstraintType.EQUALITY if equality
                                 else cooper.ConstraintType.INEQUALITY),
                formulation_type=cooper.formulations.Lagrangian,
                # DenseMultiplier's dtype defaults to float32 and ignores
                # torch.set_default_dtype. Worse, initialize_weight does
                # init.to(dtype=dtype), so that default also silently
                # *downcasts* an explicitly float64 init. Under use_float64()
                # this raises "expected Float but found Double" from the dual
                # einsum -- or, if the primal dtype happened to match, would
                # quietly cap agreement near 1e-8 and read as an algorithmic
                # discrepancy. Both have to be passed.
                multiplier=cooper.multipliers.DenseMultiplier(
                    init=torch.full((problem.m,), init,
                                    dtype=torch.get_default_dtype()),
                    dtype=torch.get_default_dtype(),
                ),
            )

        def compute_cmp_state(self, params):
            return cooper.CMPState(
                loss=problem.objective([params]),
                observed_constraints={
                    self.constraint: cooper.ConstraintState(
                        violation=problem.constraints([params]))
                },
            )

        def compute_violations(self, params):
            # AlternatingPrimalDual's dual step needs c(x_{t+1}) and nothing
            # else. Implementing this hook is Cooper's documented way to get it
            # without recomputing the loss or building a primal graph, and is
            # what keeps O1 from timing that optimizer at a handicap it does not
            # have to run under.
            return cooper.CMPState(observed_constraints={
                self.constraint: cooper.ConstraintState(
                    violation=problem.constraints([params]))})

    cmp = Wrapped()
    return x, cmp, torch.optim.SGD([x], lr=PRIMAL_LR), torch.optim.SGD(
        cmp.dual_parameters(), lr=ALM_DUAL_LR, maximize=True)


def _step_cooper(state) -> None:
    """One step. ``roll`` does its own ``zero_grad``, both backwards, both steps."""
    x, _, optimizer = state
    kwargs = {"compute_cmp_state_kwargs": {"params": x}}
    if isinstance(optimizer, cooper.optim.AlternatingPrimalDualOptimizer):
        kwargs["compute_violations_kwargs"] = {"params": x}
    optimizer.roll(**kwargs)


# --------------------------------------------------------------------------- #
# shared machinery
# --------------------------------------------------------------------------- #


def _build_cooper(problem: Problem, cooper_class, *, init=0.0, equality=False):
    """``_setup_cooper`` plus the constrained optimizer, as one runnable state."""
    x, cmp, primal, dual = _setup_cooper(problem, init=init, equality=equality)
    optimizer = cooper_class(cmp=cmp, primal_optimizers=primal,
                             dual_optimizers=dual)
    return x, cmp, optimizer


def _cooper_duals(cmp) -> torch.Tensor:
    return cmp.constraint.multiplier.weight.detach()


def _pairings():
    """The (ours, Cooper) pairings: three matched, two controls.

    One control per matched pairing whose ``ours`` side could otherwise be
    mistaken for another: the fused entry point against ``Simultaneous``, and the
    re-evaluated-constraint ordering against ``AlternatingDualPrimal``. The claim
    id is ``None`` for a control.
    """
    return [
        ("forward_update <-> AlternatingDualPrimal", DUAL_PRIMAL,
         cooper.optim.AlternatingDualPrimalOptimizer, "X1"),
        ("split forward/update <-> Simultaneous", SIMULTANEOUS,
         cooper.optim.SimultaneousOptimizer, "X2"),
        ("split + re-evaluated c <-> AlternatingPrimalDual", PRIMAL_DUAL,
         cooper.optim.AlternatingPrimalDualOptimizer, "X7"),
        ("forward_update <-> Simultaneous (control)", DUAL_PRIMAL,
         cooper.optim.SimultaneousOptimizer, None),
        ("split + re-evaluated c <-> AlternatingDualPrimal (control)", PRIMAL_DUAL,
         cooper.optim.AlternatingDualPrimalOptimizer, None),
    ]


def _problems() -> list[tuple[Problem, bool]]:
    """``(problem, equality)``. The last exercises both libraries' equality path."""
    return [
        (qp_active(), False),
        (qp_inactive(), False),
        (qp_equality_reduced(), False),
        (svm_iris(), False),
        (qp_equality(), True),
    ]


def _relative_kkt(problem: Problem, x: torch.Tensor,
                  duals: torch.Tensor) -> dict:
    """E0a's scale-free residual, so one tolerance covers every problem."""
    f, c = problem.objective([x]), problem.constraints([x])
    (gradient,) = torch.autograd.grad(f + duals @ c, x)
    c = c.detach()
    scale = 1.0 + (0.0 if problem.y_star is None
                   else float(np.abs(problem.y_star).max()))
    parts = [float(gradient.abs().max()) / scale,
             float(c.clamp(min=0.0).max())]
    gap = float("nan")
    if problem.y_star is not None:
        y_star = torch.as_tensor(problem.y_star, dtype=duals.dtype)
        gap = float((duals - y_star).abs().max())
        parts.append(gap / scale)
    return {"relative KKT": max(parts), "||y-y*||inf": gap}


def _code_lines(*functions) -> int:
    """Non-blank, non-comment, non-docstring source lines of ``functions``.

    Measured rather than asserted, because "how much does a user have to write"
    is the whole point of O2 and an eyeballed number would be worth nothing.
    """
    lines = 0
    for function in functions:
        source = textwrap.dedent(inspect.getsource(function))
        text = source.splitlines()
        body = ast.parse(source).body[0].body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]                              # drop the docstring
        numbers = set()
        for node in body:
            numbers.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        lines += sum(1 for n in sorted(numbers)
                     if text[n - 1].strip() and not text[n - 1].strip().startswith("#"))
    return lines


# --------------------------------------------------------------------------- #
# X -- parity
# --------------------------------------------------------------------------- #


def parity(problem: Problem, ordering: str, cooper_class, *,
           steps: int = PARITY_STEPS, init: float = 0.0,
           equality: bool = False) -> tuple[dict, list[dict]]:
    """Run both libraries in lockstep from a common start point.

    The two runs share their start point by *construction* -- one
    ``make_params`` result, cloned -- rather than by seeding twice and trusting
    the two draws to agree.
    """
    set_seed(0)
    ours = _setup_ours(problem, init=init, equality=equality)
    theirs = _build_cooper(problem, cooper_class, init=init, equality=equality)
    theirs[0].data.copy_(ours[0].detach())

    worst_x = worst_y = 0.0
    bitwise = bitwise_x = True
    rows = []
    for step in range(steps):
        _step_ours(problem, ours, ordering=ordering)
        _step_cooper(theirs)

        y_ours, y_theirs = ours[1].duals.detach(), _cooper_duals(theirs[1])
        dy = float((y_ours - y_theirs).abs().max())
        dx = float((ours[0].detach() - theirs[0].detach()).abs().max())
        bitwise &= torch.equal(y_ours, y_theirs)
        bitwise_x &= torch.equal(ours[0].detach(), theirs[0].detach())
        worst_y, worst_x = max(worst_y, dy), max(worst_x, dx)
        rows.append({"problem": problem.name, "step": step + 1,
                     "|dy|inf": dy, "|dx|inf": dx})

    # Exact-zero agreement: relu (Cooper) vs clamp_ (ours) must pin the *same*
    # entries to exactly 0.0.
    y_ours, y_theirs = ours[1].duals.detach(), _cooper_duals(theirs[1])
    zeros_ours = (y_ours == 0.0)
    zeros_theirs = (y_theirs == 0.0)
    return {
        "problem": problem.name,
        "m": problem.m,
        "steps": steps,
        "constraint evals/step": EVALUATIONS[ordering],
        "max |dy|inf": worst_y,
        "max |dx|inf": worst_x,
        "duals bitwise identical": bitwise,
        # Recorded rather than folded into the tolerance: the primal sides run
        # different kernels (our `snapshot @ c` against Cooper's einsum), so
        # bitwise was not expected here and is worth reporting when it holds.
        "params bitwise identical": bitwise_x,
        "exact zeros (ours)": int(zeros_ours.sum()),
        "exact zeros (cooper)": int(zeros_theirs.sum()),
        "zero sets identical": bool(torch.equal(zeros_ours, zeros_theirs)),
    }, rows


def register_parity(checks: Checks) -> tuple[list[dict], list[dict]]:
    rows, trajectories = [], []
    for name, ordering, cooper_class, claim in _pairings():
        for problem, equality in _problems():
            summary, steps = parity(problem, ordering, cooper_class,
                                    equality=equality)
            summary["pairing"] = name
            rows.append(summary)
            if problem.name == "qp_active" and claim in ("X1", "X7", None):
                # One control is enough for the figure; the second would draw on
                # top of the first.
                label = {"X1": "matched (dual-primal)",
                         "X7": "matched (primal-dual)"}.get(claim, "control")
                if not any(r["pairing"] == label for r in trajectories):
                    trajectories += [{**r, "pairing": label} for r in steps]

            print(f"  {name[:52]:<54} {problem.name:<20} "
                  f"|dy|={summary['max |dy|inf']:.2e}  "
                  f"|dx|={summary['max |dx|inf']:.2e}  "
                  f"bitwise={summary['duals bitwise identical']}")

            if claim is None:
                checks.expect(
                    max(summary["max |dy|inf"], summary["max |dx|inf"]) > CONTROL_FLOOR,
                    f"X3 control: {name} is distinguishable on {problem.name}",
                    f"max |dy|={summary['max |dy|inf']:.3e}, "
                    f"max |dx|={summary['max |dx|inf']:.3e}, floor "
                    f"{CONTROL_FLOOR:.1e}",
                )
            else:
                checks.expect(
                    summary["duals bitwise identical"]
                    and summary["max |dx|inf"] <= ALGEBRAIC,
                    f"{claim}: {name} agrees over {summary['steps']} steps on "
                    f"{problem.name} (duals bitwise, params <= {ALGEBRAIC:.1e})",
                    f"max |dy|={summary['max |dy|inf']:.3e} "
                    f"(bitwise: {summary['duals bitwise identical']}), "
                    f"max |dx|={summary['max |dx|inf']:.3e}",
                )
    return rows, trajectories


def register_projection(checks: Checks) -> list[dict]:
    """X5: ``relu`` and ``clamp_`` must agree at exactly zero.

    Started from ``y_0 = 0`` no multiplier would ever be clamped on these
    problems, so the projection would never fire and an agreement claim about it
    would be vacuous. Starting at ``y_0 = 0.5`` on ``qp_inactive``, whose ``y*``
    has exact zeros, forces the inactive multipliers down onto the bound.
    """
    rows = []
    for problem in (qp_inactive(), svm_iris()):
        summary, _ = parity(problem, DUAL_PRIMAL,
                            cooper.optim.AlternatingDualPrimalOptimizer,
                            init=0.5)
        rows.append({"problem": problem.name,
                     "y0": 0.5,
                     "exact zeros (ours)": summary["exact zeros (ours)"],
                     "exact zeros (cooper)": summary["exact zeros (cooper)"],
                     "zero sets identical": summary["zero sets identical"],
                     "duals bitwise identical": summary["duals bitwise identical"],
                     # Context, not a gate: whether these are the *right* zeros
                     # is a question about convergence, which E0a owns. Reported
                     # because a reader seeing "4 exact zeros" will ask.
                     "exact zeros in y*": int((problem.y_star == 0).sum())})
        print(f"  projection {problem.name:<20} "
              f"zeros: ours={summary['exact zeros (ours)']} "
              f"cooper={summary['exact zeros (cooper)']}  "
              f"identical={summary['zero sets identical']}")
        checks.expect(
            summary["zero sets identical"] and summary["exact zeros (ours)"] > 0,
            f"X5: clamp_ and relu pin the same multipliers to exactly 0.0 on "
            f"{problem.name}",
            f"{summary['exact zeros (ours)']} exact zeros ours, "
            f"{summary['exact zeros (cooper)']} cooper, "
            f"same entries: {summary['zero sets identical']}",
        )
    return rows


# --------------------------------------------------------------------------- #
# P -- is the recommended ordering a different algorithm?
# --------------------------------------------------------------------------- #


def phase_shift(problem: Problem, *, steps: int = PARITY_STEPS,
                equality: bool = False) -> dict:
    """X8: ``AlternatingPrimalDual`` is ``forward_update``, one dual step late.

    Both orderings iterate ``y <- P(y + g*c(x_t))`` over the same sequence of
    constraint values; dual-primal folds ``c(x_0)`` in before its first primal
    step, primal-dual after it. So handing primal-dual that one term as its
    initial multiplier should erase the difference entirely -- and if it does,
    the second constraint evaluation buys nothing but the lag.

    The dual comparison is therefore made at an offset of one: the multiplier
    Cooper holds *before* roll ``t`` is the one it uses for that primal step, and
    that is what our post-update ``y_t`` has to match.
    """
    set_seed(0)
    ours = _setup_ours(problem, equality=equality)
    x0 = ours[0].detach().clone()
    with torch.no_grad():
        c0 = problem.constraints([x0]) * ALM_DUAL_LR
    init = c0 if equality else c0.clamp(min=0.0)

    theirs = _build_cooper(problem, cooper.optim.AlternatingPrimalDualOptimizer,
                           equality=equality)
    theirs[0].data.copy_(x0)
    with torch.no_grad():
        # Written in after construction rather than through _setup_cooper, whose
        # line count O2 reports: this init is the experiment's device, not
        # something a user of either library writes.
        theirs[1].constraint.multiplier.weight.copy_(init)

    worst_shifted = worst_same = worst_x = 0.0
    bitwise_shifted = bitwise_x = True
    for _ in range(steps):
        pending = _cooper_duals(theirs[1]).clone()      # what their primal step uses
        _step_ours(problem, ours, ordering=DUAL_PRIMAL)
        _step_cooper(theirs)

        y_ours = ours[1].duals.detach()
        bitwise_shifted &= torch.equal(y_ours, pending)
        worst_shifted = max(worst_shifted, float((y_ours - pending).abs().max()))
        # The same-index gap is the control: if it were also zero the offset
        # would be doing no work and the claim would be vacuous.
        worst_same = max(worst_same,
                         float((y_ours - _cooper_duals(theirs[1])).abs().max()))
        dx = float((ours[0].detach() - theirs[0].detach()).abs().max())
        bitwise_x &= torch.equal(ours[0].detach(), theirs[0].detach())
        worst_x = max(worst_x, dx)

    return {
        "problem": problem.name,
        "m": problem.m,
        "steps": steps,
        "max |dy|inf (offset 1)": worst_shifted,
        "max |dy|inf (same index)": worst_same,
        "max |dx|inf": worst_x,
        "duals bitwise identical (offset 1)": bitwise_shifted,
        "params bitwise identical": bitwise_x,
    }


def register_phase_shift(checks: Checks) -> list[dict]:
    rows = []
    for problem, equality in _problems():
        row = phase_shift(problem, equality=equality)
        rows.append(row)
        print(f"  {problem.name:<20} |dy|(offset 1)="
              f"{row['max |dy|inf (offset 1)']:.2e} "
              f"(same index {row['max |dy|inf (same index)']:.2e})  "
              f"|dx|={row['max |dx|inf']:.2e}  "
              f"bitwise={row['duals bitwise identical (offset 1)']}")
        checks.expect(
            row["duals bitwise identical (offset 1)"]
            and row["max |dx|inf"] <= ALGEBRAIC
            and row["max |dy|inf (same index)"] > CONTROL_FLOOR,
            f"X8: on {problem.name}, AlternatingPrimalDual from "
            f"y0=[lr*c(x0)]+ is forward_update from y0=0 -- the same iterates, "
            f"one dual step out of phase",
            f"max |dy| at offset 1 = {row['max |dy|inf (offset 1)']:.3e} "
            f"(bitwise: {row['duals bitwise identical (offset 1)']}), "
            f"max |dx| = {row['max |dx|inf']:.3e}, same-index dual gap "
            f"{row['max |dy|inf (same index)']:.3e} > "
            f"{CONTROL_FLOOR:.1e}",
        )
    return rows


# --------------------------------------------------------------------------- #
# K -- KKT anchor
# --------------------------------------------------------------------------- #


def anchor(problem: Problem, *, cap: int, equality: bool = False) -> dict:
    """Both libraries to convergence or to ``cap``, each scored against ``(x*, y*)``.

    Parity alone only shows the two agree. This is what shows they agree *and*
    are right -- and that the agreement survives two orders of magnitude more
    steps than the parity horizon.

    The stopping test is *part of the comparison*, not a fixed budget. Two
    implementations of the same algorithm must not only end up at the same
    residual, they must clear the tolerance at the same iteration. That also
    keeps E0a's finding #4 out of the gate: with one fixed dual step the number
    of iterations needed scales like ``||J||^2``, so a shared budget is not equal
    progress across problems, and any per-problem exemption chosen after seeing
    the numbers would be curve-fitting.
    """
    set_seed(0)
    ours = _setup_ours(problem, equality=equality)
    theirs = _build_cooper(problem, cooper.optim.AlternatingDualPrimalOptimizer,
                           equality=equality)
    theirs[0].data.copy_(ours[0].detach())

    def residuals():
        return (_relative_kkt(problem, ours[0], ours[1].duals.detach()),
                _relative_kkt(problem, theirs[0], _cooper_duals(theirs[1])))

    solved_ours = solved_theirs = None
    bitwise = True
    ran = 0
    for step in range(1, cap + 1):
        _step_ours(problem, ours, ordering=DUAL_PRIMAL)
        _step_cooper(theirs)
        bitwise &= torch.equal(ours[1].duals.detach(), _cooper_duals(theirs[1]))
        ran = step
        if step % CHECK_EVERY and step != cap:
            continue
        ours_kkt, theirs_kkt = residuals()
        if solved_ours is None and ours_kkt["relative KKT"] <= KKT_TOLERANCE:
            solved_ours = step
        if solved_theirs is None and theirs_kkt["relative KKT"] <= KKT_TOLERANCE:
            solved_theirs = step
        if solved_ours is not None and solved_theirs is not None:
            break

    ours_kkt, theirs_kkt = residuals()
    return {
        "problem": problem.name,
        "steps run": ran,
        "cap": cap,
        "solved at (ours)": solved_ours,
        "solved at (cooper)": solved_theirs,
        "relative KKT (ours)": ours_kkt["relative KKT"],
        "relative KKT (cooper)": theirs_kkt["relative KKT"],
        "|difference|": abs(ours_kkt["relative KKT"] - theirs_kkt["relative KKT"]),
        "||y-y*||inf (ours)": ours_kkt["||y-y*||inf"],
        "||y-y*||inf (cooper)": theirs_kkt["||y-y*||inf"],
        "duals bitwise identical": bitwise,
    }


def register_anchor(checks: Checks, cap: int) -> list[dict]:
    rows = []
    for problem, equality in _problems():
        if not problem.has_reference_multipliers:
            continue
        row = anchor(problem, cap=cap, equality=equality)
        rows.append(row)
        print(f"  {problem.name:<20} relative KKT ours={row['relative KKT (ours)']:.3e} "
              f"cooper={row['relative KKT (cooper)']:.3e}  solved at "
              f"{row['solved at (ours)']}/{row['solved at (cooper)']}  "
              f"bitwise={row['duals bitwise identical']}")
        checks.expect(
            row["duals bitwise identical"]
            and row["|difference|"] <= ALGEBRAIC
            and row["solved at (ours)"] == row["solved at (cooper)"],
            f"X4: both libraries follow the same trajectory on {problem.name} "
            f"over {row['steps run']} steps, reaching the same residual with the "
            f"same {KKT_TOLERANCE:g} convergence outcome",
            f"ours {row['relative KKT (ours)']:.3e} vs cooper "
            f"{row['relative KKT (cooper)']:.3e} (difference "
            f"{row['|difference|']:.3e}), solved at "
            f"{row['solved at (ours)']} vs {row['solved at (cooper)']}, "
            f"duals bitwise: {row['duals bitwise identical']}",
        )

    # The anchoring claim, stated once and without naming a problem: unless some
    # problem is actually driven to the exact reference solution, every agreement
    # above could be agreement in being wrong together. Which problems converge
    # under one fixed step size is E0a's question, not this script's.
    solved = [r["problem"] for r in rows if r["solved at (ours)"] is not None]
    checks.expect(
        bool(solved),
        f"X4: the shared trajectory reaches the exact (x*, y*) on at least one "
        f"problem, so the agreement is not agreement in being wrong",
        f"solved to {KKT_TOLERANCE:g} within {cap} steps: "
        f"{', '.join(solved) or 'none'}; residuals elsewhere "
        + ", ".join(f"{r['problem']} {r['relative KKT (ours)']:.1e}"
                    for r in rows if r["solved at (ours)"] is None),
    )
    return rows


def orderings(problem: Problem, *, cap: int, equality: bool = False) -> list[dict]:
    """X9: the three orderings from a common ``y_0 = 0``, each to the tolerance.

    Run on our side only, which X1/X2/X7 licenses: each ordering is bitwise its
    Cooper counterpart, so this is a statement about both libraries at a third of
    the wall clock. ``constraint evals`` is the column that matters -- iterations
    are not the price primal-dual actually pays.
    """
    rows = []
    for ordering in (DUAL_PRIMAL, SIMULTANEOUS, PRIMAL_DUAL):
        set_seed(0)
        state = _setup_ours(problem, equality=equality)
        solved, ran = None, 0
        for step in range(1, cap + 1):
            _step_ours(problem, state, ordering=ordering)
            ran = step
            if step % CHECK_EVERY and step != cap:
                continue
            if _relative_kkt(problem, state[0],
                             state[1].duals.detach())["relative KKT"] <= KKT_TOLERANCE:
                solved = step
                break
        kkt = _relative_kkt(problem, state[0], state[1].duals.detach())
        rows.append({
            "problem": problem.name,
            "ordering": ordering,
            "cooper counterpart": {
                DUAL_PRIMAL: "AlternatingDualPrimal",
                SIMULTANEOUS: "Simultaneous",
                PRIMAL_DUAL: "AlternatingPrimalDual"}[ordering],
            "constraint evals/step": EVALUATIONS[ordering],
            "steps run": ran,
            "solved at": solved,
            "constraint evals to tolerance":
                None if solved is None else solved * EVALUATIONS[ordering],
            "relative KKT": kkt["relative KKT"],
            "||y-y*||inf": kkt["||y-y*||inf"],
        })
    return rows


def _no_better(pd: dict, dp: dict) -> bool:
    """Whether the primal-dual row failed to get ahead of the dual-primal one.

    Solved-earlier beats solved-later beats not-solved; among rows that never
    cleared the tolerance, the lower residual wins. Written out rather than
    scored on ``solved at`` alone, because ``None`` is not a large number.
    """
    if pd["solved at"] is None and dp["solved at"] is None:
        return pd["relative KKT"] >= dp["relative KKT"]
    if pd["solved at"] is None:
        return True
    if dp["solved at"] is None:
        return False
    return pd["solved at"] >= dp["solved at"]


def register_orderings(checks: Checks, cap: int) -> list[dict]:
    rows = []
    for problem, equality in _problems():
        if not problem.has_reference_multipliers:
            continue
        group = orderings(problem, cap=cap, equality=equality)
        rows += group
        by_ordering = {row["ordering"]: row for row in group}
        for row in group:
            print(f"  {problem.name:<20} {row['ordering']:<14} "
                  f"solved at {str(row['solved at']):>6} "
                  f"({str(row['constraint evals to tolerance']):>6} constraint "
                  f"evals)  relative KKT={row['relative KKT']:.3e}")
        pd, dp = by_ordering[PRIMAL_DUAL], by_ordering[DUAL_PRIMAL]
        checks.expect(
            _no_better(pd, dp),
            f"X9: on {problem.name} the primal-dual ordering is not ahead of "
            f"dual-primal, at twice the constraint evaluations per step",
            f"solved at {pd['solved at']} (primal-dual, "
            f"{pd['constraint evals to tolerance']} constraint evals) vs "
            f"{dp['solved at']} (dual-primal, "
            f"{dp['constraint evals to tolerance']} constraint evals); "
            f"final relative KKT {pd['relative KKT']:.3e} vs "
            f"{dp['relative KKT']:.3e}",
        )
    return rows


# --------------------------------------------------------------------------- #
# S -- a stochastic, data-dependent constraint
# --------------------------------------------------------------------------- #


def _setup_ours_fair(problem, model):
    dual = ALM(m=problem.m, lr=FAIR_DUAL_LR, penalty=0.0, init_duals=0.0,
               is_ineq=True, dual_range=(0.0, math.inf),
               momentum=0.0, dampening=0.0, restart=False)
    return dual, torch.optim.Adam(model.parameters(), lr=FAIR_PRIMAL_LR)


def _step_ours_fair(problem, model, dual, primal, batch) -> None:
    features, sens, labels = batch
    logits = model(features)
    loss = problem.objective(logits, labels)
    primal.zero_grad()
    dual.forward_update(loss, problem.constraints(logits, sens)).backward()
    primal.step()


def _setup_cooper_fair(problem, model):
    class Wrapped(cooper.ConstrainedMinimizationProblem):
        def __init__(self):
            super().__init__()
            self.constraint = cooper.Constraint(
                constraint_type=cooper.ConstraintType.INEQUALITY,
                formulation_type=cooper.formulations.Lagrangian,
                multiplier=cooper.multipliers.DenseMultiplier(
                    num_constraints=problem.m, dtype=torch.get_default_dtype()),
            )

        def compute_cmp_state(self, batch):
            features, sens, labels = batch
            logits = model(features)
            return cooper.CMPState(
                loss=problem.objective(logits, labels),
                observed_constraints={
                    self.constraint: cooper.ConstraintState(
                        violation=problem.constraints(logits, sens))
                },
            )

    cmp = Wrapped()
    return cmp, cooper.optim.AlternatingDualPrimalOptimizer(
        cmp=cmp,
        primal_optimizers=torch.optim.Adam(model.parameters(), lr=FAIR_PRIMAL_LR),
        dual_optimizers=torch.optim.SGD(cmp.dual_parameters(), lr=FAIR_DUAL_LR,
                                        maximize=True),
    )


def register_stochastic(checks: Checks, steps: int) -> list[dict]:
    """X6: parity when the constraint is a minibatch estimate of an expectation.

    Everything rests on both runs seeing the *same* batches and the *same*
    initial weights, so both are built once and replayed -- E2a's
    shared-generator bug says a comparison that merely trusts two shufflers is
    worth nothing.
    """
    from paper.problems import fairness

    if not fairness.available_states():
        print("  ACS data not on disk -- skipping the fairness parity check")
        return []

    problem = fairness.build(dataset="income", shape="pairwise", bound=FAIR_BOUND)
    problem.reseed(0)
    batches = []
    while len(batches) < steps:
        for batch in problem.loader:
            batches.append(batch)
            if len(batches) >= steps:
                break

    set_seed(0)
    model_ours = problem.make_model()
    model_theirs = copy.deepcopy(model_ours)

    dual, primal = _setup_ours_fair(problem, model_ours)
    cmp, optimizer = _setup_cooper_fair(problem, model_theirs)

    worst_y = worst_w = 0.0
    bitwise = True
    for batch in batches:
        _step_ours_fair(problem, model_ours, dual, primal, batch)
        optimizer.roll(compute_cmp_state_kwargs={"batch": batch})

        y_ours, y_theirs = dual.duals.detach(), _cooper_duals(cmp)
        bitwise &= torch.equal(y_ours, y_theirs)
        worst_y = max(worst_y, float((y_ours - y_theirs).abs().max()))
        worst_w = max(worst_w, max(
            float((a.detach() - b.detach()).abs().max())
            for a, b in zip(model_ours.parameters(), model_theirs.parameters())))

    row = {"problem": problem.name, "m": problem.m, "steps": steps,
           "batch size": problem.batch_size,
           "max |dy|inf": worst_y, "max |dweights|inf": worst_w,
           "duals bitwise identical": bitwise, "notes": problem.notes}
    print(f"  {problem.name:<20} m={problem.m} |dy|={worst_y:.3e} "
          f"|dw|={worst_w:.3e} bitwise={bitwise}")
    checks.expect(
        bitwise and worst_w <= ALGEBRAIC,
        f"X6: identical trajectories on {problem.name} (m={problem.m}, "
        f"minibatch-stochastic constraint) over {steps} steps",
        f"max |dy|={worst_y:.3e} (bitwise: {bitwise}), "
        f"max |dweights|={worst_w:.3e}",
    )
    return [row]


# --------------------------------------------------------------------------- #
# O -- observations
# --------------------------------------------------------------------------- #


def overhead(steps: int = OVERHEAD_STEPS) -> list[dict]:
    """Per-step cost of each dual layer. Both are pure Python, so this is fair.

    Unlike E2a's Adam comparison, neither side is paying for a different
    constraint evaluation: the same ``problem.constraints`` closure runs in both.
    """
    rows = []
    for problem, equality in _problems():
        timings = {}

        for label, ordering in (("ours (dual-primal)", DUAL_PRIMAL),
                                ("ours (primal-dual)", PRIMAL_DUAL)):
            set_seed(0)
            ours = _setup_ours(problem, equality=equality)
            for _ in range(50):
                _step_ours(problem, ours, ordering=ordering)
            start = time.perf_counter()
            for _ in range(steps):
                _step_ours(problem, ours, ordering=ordering)
            timings[label] = (time.perf_counter() - start) / steps

        for label, cooper_class in (
                ("cooper (AlternatingDualPrimal)",
                 cooper.optim.AlternatingDualPrimalOptimizer),
                ("cooper (AlternatingPrimalDual)",
                 cooper.optim.AlternatingPrimalDualOptimizer)):
            set_seed(0)
            theirs = _build_cooper(problem, cooper_class, equality=equality)
            for _ in range(50):
                _step_cooper(theirs)
            start = time.perf_counter()
            for _ in range(steps):
                _step_cooper(theirs)
            timings[label] = (time.perf_counter() - start) / steps

        row = {"problem": problem.name, "m": problem.m, "steps": steps}
        for label, seconds in timings.items():
            row[f"{label} us/step"] = seconds * 1e6
        row["ratio cooper/ours (dual-primal)"] = (
            timings["cooper (AlternatingDualPrimal)"] / timings["ours (dual-primal)"])
        row["ratio cooper/ours (primal-dual)"] = (
            timings["cooper (AlternatingPrimalDual)"] / timings["ours (primal-dual)"])
        row["ratio primal-dual/dual-primal (ours)"] = (
            timings["ours (primal-dual)"] / timings["ours (dual-primal)"])
        rows.append(row)
        print(f"  {problem.name:<20} "
              f"dual-primal ours={row['ours (dual-primal) us/step']:6.1f} "
              f"cooper={row['cooper (AlternatingDualPrimal) us/step']:6.1f}  |  "
              f"primal-dual ours={row['ours (primal-dual) us/step']:6.1f} "
              f"cooper={row['cooper (AlternatingPrimalDual) us/step']:6.1f} us")
    return rows


def api_surface() -> list[dict]:
    """O2: what a user writes, and what the extra surface buys.

    The line counts are of this file's own four adapter functions, which do the
    same job for each library, so the comparison is like-for-like. The "buys"
    column is not decoration: Cooper's heavier surface pays for capabilities
    ``dual_optim`` does not have, and a table that omitted them would be a
    strawman.
    """
    return [
        {"aspect": "user code to pose the problem + one step, all three step "
                   "orderings (non-comment lines)",
         "ours": _code_lines(_setup_ours, _step_ours),
         "cooper": _code_lines(_setup_cooper, _step_cooper),
         "what cooper's extra surface buys": "-"},
        {"aspect": "choosing the primal/dual step ordering",
         "ours": "which entry point the loop calls: forward_update; split "
                 "forward/update; split with the constraint closure called "
                 "again after primal.step()",
         "cooper": "which optimizer class is constructed: "
                   "AlternatingDualPrimal, Simultaneous, AlternatingPrimalDual",
         "what cooper's extra surface buys": "the ordering cannot be got wrong "
                                             "by writing the loop wrong, and "
                                             "compute_violations lets the "
                                             "primal-dual re-evaluation skip the "
                                             "loss and the primal graph"},
        {"aspect": "objects a user constructs",
         "ours": "1 (ALM) + own torch optimizer + own loop",
         "cooper": "CMP subclass, Constraint, Multiplier, CMPState, "
                   "ConstraintState, ConstrainedOptimizer",
         "what cooper's extra surface buys": "formulation and multiplier are "
                                             "swappable independently of the "
                                             "optimizer"},
        {"aspect": "who owns the training loop",
         "ours": "the user (dual layer is a torch.optim.Optimizer over duals)",
         "cooper": "the library (optimizer.roll drives forward, both backwards, "
                   "both steps)",
         "what cooper's extra surface buys": "primal/dual step ordering is a "
                                             "class choice, not an idiom the "
                                             "user has to get right"},
        {"aspect": "sparse / per-sample multipliers",
         "ours": "no", "cooper": "IndexedMultiplier, ImplicitMultiplier, "
                                 "constraint_features",
         "what cooper's extra surface buys": "one multiplier per constraint "
                                             "instance at dataset scale"},
        {"aspect": "separating the differentiable surrogate from the measurement "
                   "that drives the multiplier",
         "ours": "no", "cooper": "ConstraintState.strict_violation",
         "what cooper's extra surface buys": "duals can follow an exact or "
                                             "non-differentiable statistic"},
        {"aspect": "constraint groups with independent dual step / bounds / bound",
         "ours": "add_constraint_group, named or positional",
         "cooper": "one Constraint attribute per group",
         "what cooper's extra surface buys": "-"},
        {"aspect": "data-parallel reduction of constraint values",
         "ours": "process_group=..., all_reduce(AVG) before the dual update",
         "cooper": "none",
         "what cooper's extra surface buys": "-"},
        {"aspect": "dual restart (arXiv:2208.04425 Eq. 6)",
         "ours": "restart=True", "cooper": "not in 1.0.1 (0.x had dual_restarts)",
         "what cooper's extra surface buys": "-"},
        {"aspect": "penalty coefficient schedules",
         "ours": "iALM's sigma; PBM's six penalty rules",
         "cooper": "PenaltyCoefficientUpdater (multiplicative, additive, "
                   "feasibility-driven)",
         "what cooper's extra surface buys": "schedule is composable with any "
                                             "formulation"},
    ]


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def make_figure(trajectories: list[dict], shifted: list[dict]) -> None:
    """Left: matched vs mismatched pairings. Right: X8, the one-step offset."""
    fig, axes, plt = figure(1, 2, row_height=2.6)
    # Bitwise agreement is exactly 0 and cannot be drawn on a log axis; the floor
    # makes it visible as "at rounding" rather than dropping the line.
    floor = ALGEBRAIC * 1e-2

    for pairing, style in (
            ("matched (dual-primal)", {"color": "#2E86AB", "ls": "-"}),
            ("matched (primal-dual)", {"color": "#3E8E5A", "ls": "-.", "lw": 1.6}),
            ("control", {"color": "#D1495B", "ls": "--"})):
        rows = [r for r in trajectories if r["pairing"] == pairing]
        if not rows:
            continue
        axes[0].plot([r["step"] for r in rows],
                     [max(r["|dy|inf"], floor) for r in rows],
                     label=pairing, **style)
    axes[0].set_title("ALM(penalty=0) vs Cooper: three matched orderings, "
                      "one control")
    axes[0].set_ylabel(r"$\|y_{ours} - y_{cooper}\|_\infty$")

    row = next((r for r in shifted if r["problem"] == "qp_active"), None)
    if row is not None:
        # A two-bar summary rather than a trajectory: the claim is about two
        # numbers, one of which is exactly zero for every step.
        axes[1].bar(["offset 1", "same index"],
                    [max(row["max |dy|inf (offset 1)"], floor),
                     row["max |dy|inf (same index)"]],
                    color=["#2E86AB", "#D1495B"], width=0.55)
        axes[1].set_yscale("log")
        # The left bar is drawn at the floor but *is* zero; unlabelled it reads
        # as "small", which is a weaker claim than the one being made.
        axes[1].annotate("exactly 0", (0, floor), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=6)
        axes[1].set_ylabel(r"$\max_t \|y_t^{ours} - y^{cooper}\|_\infty$")
        axes[1].set_title("AlternatingPrimalDual from "
                          r"$y_0=[\eta c(x_0)]_+$" "\nis forward_update, "
                          "one dual step late")

    for ax in axes:
        ax.axhline(ALGEBRAIC, color="0.5", lw=0.8, ls=":",
                   label="algebraic tolerance")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
    axes[0].set_xlabel("step")
    fig.suptitle("qp_active", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "e0e_parity", EXPERIMENT)
    plt.close(fig)


# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="E0e: cross-validation of ALM(penalty=0) against Cooper")
    parser.add_argument("--quick", action="store_true",
                        help="parity only -- the algebra, in seconds")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--anchor-cap", type=int, default=ANCHOR_CAP,
                        help="iteration cap for the KKT anchor")
    parser.add_argument("--fair-steps", type=int, default=FAIR_STEPS)
    args = parser.parse_args(argv)

    if not HAVE_COOPER:
        # Not a failure: cooper is an optional [compare] extra, and run_all.sh
        # must stay green on a machine without it.
        print("cooper is not installed -- skipping E0e "
              "(pip install 'cooper-optim>=1.0.1')")
        sys.exit(0)

    use_float64()
    checks = Checks(enabled=args.check)
    print(f"cooper {cooper.__version__}, torch {torch.__version__}")
    versions = {"cooper": cooper.__version__, "torch": torch.__version__}

    # ---- X ---------------------------------------------------------------- #
    print("\nX — parity of the two libraries, matched and mismatched orderings")
    parity_rows, trajectories = register_parity(checks)
    write_table([{**r, **versions} for r in parity_rows], "e0e_parity", EXPERIMENT,
                columns=["pairing", "problem", "m", "steps",
                         "constraint evals/step", "max |dy|inf",
                         "max |dx|inf", "duals bitwise identical",
                         "params bitwise identical", "cooper", "torch"],
                title="E0e/X: ALM(penalty=0) against Cooper's Lagrangian + "
                      "SGD-ascent multiplier, 200 steps from a common start. Our "
                      "forward_update uses post-update multipliers, so it pairs "
                      "with AlternatingDualPrimalOptimizer; the split API pairs "
                      "with SimultaneousOptimizer; and the split API with the "
                      "constraints re-evaluated after primal.step() pairs with "
                      "AlternatingPrimalDualOptimizer, the one Cooper's docs "
                      "recommend -- reached without a new entry point, at the "
                      "cost of a second constraint evaluation per step. The last "
                      "two pairings are the controls that make the first three "
                      "non-vacuous. Only the duals were expected to agree "
                      "bitwise; that the parameters do as well says torch's "
                      "einsum and matmul reductions coincide at these sizes, "
                      "which is a fact about the kernels rather than about "
                      "either library.")

    print("\nX5 — clamp_ vs relu at exactly zero")
    projection_rows = register_projection(checks)
    write_table([{**r, **versions} for r in projection_rows],
                "e0e_projection", EXPERIMENT,
                title="E0e/X5: started from y0=0.5 so the projection actually "
                      "fires. Our clamp_ and Cooper's relu must pin the same "
                      "multipliers to exactly 0.0.")

    print("\nP — is Cooper's recommended ordering a different algorithm?")
    shifted_rows = register_phase_shift(checks)
    write_table([{**r, **versions} for r in shifted_rows], "e0e_phase_shift",
                EXPERIMENT,
                title="E0e/X8: AlternatingPrimalDualOptimizer started from "
                      "y0 = [lr*c(x0)]+ against forward_update started from "
                      "y0 = 0. Both orderings iterate y <- P(y + lr*c(x_t)) over "
                      "the same constraint sequence and differ only in whether "
                      "c(x_0) is folded in before or after the first primal "
                      "step, so handing primal-dual that one term erases the "
                      "difference: identical parameters, and duals that are "
                      "bitwise identical at an offset of one step. The "
                      "same-index column is the control -- were it also zero the "
                      "offset would be doing no work. The recommended optimizer "
                      "is therefore not a different algorithm but the same one "
                      "lagging by a dual step, and it pays a second constraint "
                      "evaluation per step for the lag.")

    write_csv(trajectories, "e0e_trajectories", EXPERIMENT)
    make_figure(trajectories, shifted_rows)

    if args.quick:
        main_exit(checks, EXPERIMENT, "e0e_predictions")
        return

    # ---- K ---------------------------------------------------------------- #
    print(f"\nK — KKT anchor, to convergence or {args.anchor_cap} steps")
    anchor_rows = register_anchor(checks, args.anchor_cap)
    write_table([{**r, **versions} for r in anchor_rows], "e0e_kkt", EXPERIMENT,
                title=f"E0e/K: both libraries run until the relative KKT residual "
                      f"clears {KKT_TOLERANCE:g} or to {args.anchor_cap} steps, "
                      f"scored against the exact (x*, y*). This is what separates "
                      f"'agree with each other' from 'agree and are right'. The "
                      f"gate is agreement plus an identical convergence "
                      f"iteration, so no fixed budget has to be defended; the "
                      f"residual a given problem reaches under one untuned step "
                      f"size is E0a's subject, and the spread here is its finding "
                      f"#4 again -- svm_iris (m=100) gets further than qp_active "
                      f"(m=5), because progress tracks the dual step against "
                      f"||J||^2 rather than problem size.")

    print(f"\nK — the three orderings from y0 = 0, to convergence or "
          f"{args.anchor_cap} steps")
    ordering_rows = register_orderings(checks, args.anchor_cap)
    write_table([{**r, **versions} for r in ordering_rows], "e0e_orderings",
                EXPERIMENT,
                title=f"E0e/X9: the practical reading of X8. Each ordering runs "
                      f"from a common y0 = 0 until the relative KKT residual "
                      f"clears {KKT_TOLERANCE:g} or to {args.anchor_cap} steps. "
                      f"Run on our side only, which X1/X2/X7 licenses: every "
                      f"ordering is bitwise its Cooper counterpart, so this is a "
                      f"statement about both libraries. The column that decides "
                      f"the question is 'constraint evals to tolerance', not "
                      f"'solved at': primal-dual re-evaluates the constraints at "
                      f"the new iterate, so an equal iteration count is twice "
                      f"the work.")

    # ---- S ---------------------------------------------------------------- #
    print("\nS — a minibatch-stochastic, data-dependent constraint")
    fair_rows = register_stochastic(checks, args.fair_steps)
    if fair_rows:
        write_table([{**r, **versions} for r in fair_rows], "e0e_stochastic",
                    EXPERIMENT,
                    columns=["problem", "m", "steps", "batch size",
                             "max |dy|inf", "max |dweights|inf",
                             "duals bitwise identical", "cooper", "torch",
                             "notes"],
                    title="E0e/X6: E2a's income_pairwise problem, identical "
                          "batches and identical initial weights by "
                          "construction. Run in float64 here, unlike E2a's "
                          "float32 pipeline, so the script keeps one precision "
                          "policy and the bar can stay at rounding.")

    # ---- O ---------------------------------------------------------------- #
    print("\nO — observations (reported, not gated)")
    overhead_rows = overhead()
    write_table([{**r, **versions} for r in overhead_rows], "e0e_overhead",
                EXPERIMENT,
                title="E0e/O1: per-step wall clock. Both dual layers are pure "
                      "Python and run the identical constraint closure, so this "
                      "is a like-for-like number -- unlike E2a's Adam baseline, "
                      "which never evaluates the constraint at all. Absolute "
                      "microseconds, since at these problem sizes a ratio is "
                      "dominated by interpreter overhead. The ratio has a "
                      "mechanism rather than being a mystery: Cooper builds two "
                      "scalars from one forward and calls backward on each, "
                      "which its own source notes as going over the constraints "
                      "twice (simultaneous_optimizer.py:60-62), where our "
                      "surrogate is one scalar and one backward. On these "
                      "problems the model is small enough that this is the whole "
                      "step; at LM scale it would be invisible either way, which "
                      "is E3's measurement, not this one's. The primal-dual "
                      "columns carry the second constraint evaluation on both "
                      "sides -- Cooper's through the compute_violations hook, "
                      "its documented way to skip the loss and the primal graph, "
                      "so neither library is timed at a handicap it does not "
                      "have to run under. This is the one E0e artifact that is "
                      "not byte-for-byte reproducible, since it measures wall "
                      "clock; every other table and the CSV are.")
    api_rows = api_surface()
    write_table(api_rows, "e0e_api", EXPERIMENT,
                title="E0e/O2: API surface. Line counts are of this script's own "
                      "four adapter functions, which do the same job for each "
                      "library across all three step orderings -- including, on "
                      "the Cooper side, the compute_violations hook its "
                      "primal-dual roll needs to avoid a redundant loss and "
                      "graph. The third column records what Cooper's larger "
                      "surface buys, because a table without it would be a "
                      "strawman.")
    for row in api_rows[:1]:
        print(f"  user code: ours {row['ours']} lines, cooper {row['cooper']} lines")

    main_exit(checks, EXPERIMENT, "e0e_predictions")


if __name__ == "__main__":
    main()
