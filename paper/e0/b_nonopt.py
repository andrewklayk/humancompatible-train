"""
E0b — does the ``NonOpt`` port reproduce published values on the standard
nonsmooth test set?

``sqp/nonopt`` is a PyTorch port of Curtis's C++ NonOpt. Its algorithms are not
ours, so the paper owes a check that the port solves what the original solves.
Building the C++ reference is deferred; instead this compares against the
*published* values for the same ten problems at ``n = 50``: ``f(x0)`` and
``f(x*)`` are tabulated, so "the port reaches the known optimal value from the
prescribed starting point" is directly falsifiable.

Two independent things are being measured, and they should not be conflated:

1. **Does it reach ``f*``?** Reported as the relative gap
   ``(f - f*)/max(1, |f*|)``. This is the faithfulness claim.
2. **What do our defaults cost?** Our defaults differ from the reference
   configuration in two ways we can measure, so we do:

   * ``inverse_hessian``: our default is ``limited_memory``; the reference used
     full-memory BFGS, and Curtis & Que state plainly that limited memory "do[es]
     not typically perform as well as a full memory approach in the context of
     nonsmooth optimization". So ``dense`` is the reference-matching setting and
     our own default is a deviation to quantify.
   * ``point_set_options["size_factor"]``: pruning keeps
     ``int(min(size_factor * n, size_maximum))`` points, which at ``n = 50`` with
     the default ``0.05`` is **two** — and for ``n < 20`` it is **zero**, i.e. the
     bundle is emptied every step. This is likely the largest single deviation
     from the reference at this dimension, so ``size_factor`` is swept.

Three implementation facts, all found by reading the code, are worked around here
rather than papered over — and they are reported, since a user of the package will
hit them too:

* **There are no evaluation counters.** The closure is wrapped to count calls. One
  closure call always produces both the value and the gradient, so ``#func`` and
  ``#grad`` cannot be separated the way the published tables do; the count below
  is closure calls, which upper-bounds both.
* **``step()`` returns the loss only on the first call** (it returns ``None``
  afterwards). Trajectories must be read from ``optimizer.state[p]["f"]``.
* **The stationarity test scales with the initial gradient.** The effective test
  is ``||G omega||_inf <= tol * max(1, ||g0||_inf)``, and
  ``stationarity_reference`` is set once at initialisation and never updated, so
  on ``maxq`` (``||g0||_inf = 100``) the effective tolerance is ``1e-2``, not the
  requested ``1e-4``. Both the requested and the effective tolerance are reported.

Usage::

    python paper/e0/b_nonopt.py                 # full grid
    python paper/e0/b_nonopt.py --quick         # one configuration, 200 iterations
    python paper/e0/b_nonopt.py --check         # exit non-zero on a failed prediction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from humancompatible.train.sqp import NonOpt
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
from paper.problems import nonsmooth

EXPERIMENT = "e0b"

DIMENSION = 50
# The reference's own criterion (5.1) is ||G omega||_2 <= 10*delta_k with
# delta_k <= 1e-4; ours is an inf-norm test scaled by ||g0||_inf. Requesting 1e-4
# is the closest match the interface allows.
STATIONARITY_TOLERANCE = 1e-4
MAX_ITERATIONS = 10_000

# Our three direction strategies against the reference's three SVANO variants.
# This mapping is what makes the per-strategy comparison meaningful rather than
# a comparison of arbitrary settings.
DIRECTIONS = {
    "gradient": "SVANO-BFGS",
    "cutting_plane": "SVANO-Bundle",
    "gradient_combination": "SVANO-GS",
}
INVERSE_HESSIANS = ["dense", "limited_memory"]
# 0.05 is the shipped default (two points at n=50); 0.4 gives 20; 2.0 gives 100,
# which is also the shipped size_maximum, i.e. no factor-based pruning at all.
SIZE_FACTORS = [0.05, 0.4, 2.0]


def run(problem, *, direction, inverse_hessian, size_factor,
        n=DIMENSION, max_iterations=MAX_ITERATIONS, record=False):
    """Run one configuration; return a record and, optionally, a trajectory."""
    set_seed(0)
    x = torch.nn.Parameter(problem.x0(n).clone())
    optimizer = NonOpt(
        [x],
        direction=direction,
        inverse_hessian=inverse_hessian,
        stationarity_tolerance=STATIONARITY_TOLERANCE,
        point_set_options={"size_factor": size_factor},
    )

    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = problem.objective(x)
        loss.backward()
        return loss

    state = optimizer.state[x]
    trajectory = []
    for _ in range(max_iterations):
        optimizer.step(closure)
        if record:
            # step() returns the loss only on its first call, so read the state.
            trajectory.append({"iteration": state["n_iterations"],
                               "calls": calls, "f": state["f"]})
        if optimizer.status != "running":
            break

    f_final = float(problem.objective(x.detach()))
    target = problem.target(n)
    record_row = {
        "problem": problem.name,
        "convex": problem.is_convex,
        "direction": direction,
        "reference algorithm": DIRECTIONS[direction],
        "inverse_hessian": inverse_hessian,
        "size_factor": size_factor,
        "point set limit": int(min(size_factor * n, 100)),
        "iterations": state["n_iterations"],
        "closure calls": calls,
        "status": optimizer.status,
        "f(x0)": problem.f_x0(n),
        "f final": f_final,
        "f*": target,
        # chained_mifflin_2 has no closed-form optimum, so its target is the
        # published value printed to one decimal: the gap there cannot go below
        # ~1.4e-4 no matter how well the solver does, and must not be read as
        # solver error.
        "f* source": "closed form" if problem.f_star is not None
                     else "published (1 decimal)",
        "rel gap": (f_final - target) / max(1.0, abs(target)),
        # What the solver actually enforced, as opposed to what was requested.
        "effective tol": STATIONARITY_TOLERANCE
        * max(1.0, state.get("stationarity_reference", 1.0)),
        "final radius": optimizer.stationarity_radius,
    }
    return record_row, trajectory


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #


def make_figure(trajectories: dict, problem_names: list[str]):
    """One panel per problem: ``f - f*`` against closure calls, one line per direction."""
    ncols = 5
    nrows = int(np.ceil(len(problem_names) / ncols))
    fig, axes, plt = figure(nrows, ncols, row_height=1.9)
    styles = {
        "gradient": {"color": "#2E86AB", "ls": "-"},
        "cutting_plane": {"color": "#D1495B", "ls": "--"},
        "gradient_combination": {"color": "#E0A458", "ls": ":"},
    }
    for ax, name in zip(axes, problem_names):
        for direction, style in styles.items():
            rows = trajectories.get((name, direction))
            if not rows:
                continue
            calls = [r["calls"] for r in rows]
            gaps = [max(r["gap"], 1e-16) for r in rows]
            ax.plot(calls, gaps, label=DIRECTIONS[direction], **style)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name.replace("_", " "), fontsize=7)
        ax.set_xlabel("closure calls")
    axes[0].set_ylabel(r"$f - f^\star$")
    if len(axes) > ncols:
        axes[ncols].set_ylabel(r"$f - f^\star$")
    for ax in axes[len(problem_names):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "e0b_nonopt", EXPERIMENT)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #

GAP_TOLERANCE = 1e-3


def register_predictions(checks: Checks, records: list[dict]) -> None:
    """Predictions registered before the first run.

    Q1 On every one of the ten problems, *some* configuration in the grid reaches
       the published ``f*`` to a relative gap of 1e-3. This is the faithfulness
       claim; a problem where nothing in the grid gets close is either a port
       defect or a transcription error, and the ``f(x0)`` check in
       ``problems/nonsmooth.py`` has already excluded the latter.
    Q2 On the five convex problems, the *best* configuration reaches 1e-3. The
       nonsmooth-convex case is where the theory is strongest, so a failure here
       is more serious than one on a nonconvex problem.
    Q3 ``dense`` beats ``limited_memory`` on the median relative gap, since the
       reference's own remark is that limited-memory BFGS underperforms in
       nonsmooth optimization — and if so, our shipped default is a deviation
       worth documenting.
    Q4 Raising ``size_factor`` above the shipped 0.05 improves the median gap,
       because two bundle points at n=50 is a severe truncation.
       **False, and recorded rather than edited.** The measurement stands; the
       inference behind the prediction did not. "Two points must be too few" came
       from the pruning arithmetic alone, with nothing to show that bundle size is
       what limits accuracy at this dimension — and it is not, since a larger point
       set is slightly *worse* and the shipped default wins on 8 of the 10 problems.
       This experiment does not establish why, and does not record the per-iteration
       bundle contents that would be needed to. What it does establish is that
       ``inverse_hessian`` (Q3), not ``size_factor``, is the deviation from the
       reference configuration that costs something.
    """
    by_problem = {}
    for row in records:
        by_problem.setdefault(row["problem"], []).append(row)

    # Q1 / Q2
    for name, rows in by_problem.items():
        best = min(rows, key=lambda r: abs(r["rel gap"]))
        ok = abs(best["rel gap"]) <= GAP_TOLERANCE
        detail = (f"best rel gap {best['rel gap']:.3e} via {best['direction']}/"
                  f"{best['inverse_hessian']}/size_factor={best['size_factor']}, "
                  f"status={best['status']}")
        checks.expect(ok, f"Q1: some configuration reaches f* on {name}", detail)
        if best["convex"]:
            checks.expect(ok, f"Q2: {name} is convex, so reaching f* is expected",
                          detail)

    # Q3
    def median_gap(predicate):
        gaps = [abs(r["rel gap"]) for r in records if predicate(r)]
        return float(np.median(gaps)) if gaps else float("nan")

    dense = median_gap(lambda r: r["inverse_hessian"] == "dense")
    limited = median_gap(lambda r: r["inverse_hessian"] == "limited_memory")
    checks.expect(
        dense < limited,
        "Q3: dense (the reference-matching setting) beats the shipped "
        "limited_memory default on the median relative gap",
        f"dense {dense:.3e} vs limited_memory {limited:.3e}",
    )

    # Q4
    smallest = median_gap(lambda r: r["size_factor"] == min(SIZE_FACTORS))
    largest = median_gap(lambda r: r["size_factor"] == max(SIZE_FACTORS))
    wins = {
        name: min(rows, key=lambda r: abs(r["rel gap"]))["size_factor"]
        for name, rows in by_problem.items()
    }
    at_default = sum(1 for f in wins.values() if f == min(SIZE_FACTORS))
    checks.expect(
        largest < smallest,
        f"Q4: size_factor={max(SIZE_FACTORS)} beats the shipped "
        f"{min(SIZE_FACTORS)} (which keeps only "
        f"{int(min(SIZE_FACTORS) * DIMENSION)} bundle points at n={DIMENSION})",
        f"{max(SIZE_FACTORS)}: {largest:.3e} vs {min(SIZE_FACTORS)}: {smallest:.3e}; "
        f"the shipped default wins on {at_default} of {len(wins)} problems",
        known_false=(
            "the measurement stands; the inference behind the prediction did not. "
            "'Two bundle points must be too few' was asserted from the pruning "
            "arithmetic alone, with no evidence that bundle size is what limits "
            "accuracy here — and it is not: a larger point set is slightly worse. "
            "Why is NOT established by this experiment. One untested possibility is "
            "that a larger bundle retains gradients from more distant iterates, "
            "which the direction QP then averages; establishing that would need the "
            "per-iteration bundle contents, which the experiment does not record. "
            "What the data does support: the shipped default is not the deviation "
            "from the reference configuration that costs anything -- "
            "inverse_hessian (Q3) is."
        ),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="E0b: NonOpt vs published values")
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--quick", action="store_true",
                        help="one configuration, 200 iterations")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--problems", nargs="*", default=None)
    args = parser.parse_args(argv)

    use_float64()
    n = args.dimension
    max_iterations = 200 if args.quick else args.max_iterations

    # Transcription first: a wrong formula must fail here, not look like a solver
    # failure ten thousand iterations later.
    provenance = nonsmooth.verify(n) if n == DIMENSION else nonsmooth.verify(n, 1e18)
    write_table(
        provenance, "e0b_problems", EXPERIMENT,
        title=f"E0b: the ten test problems at n = {n}, transcription verified",
    )
    print(f"transcription verified for {len(provenance)} problems at n = {n}")

    names = args.problems or list(nonsmooth.PROBLEMS)
    directions = list(DIRECTIONS) if not args.quick else ["cutting_plane"]
    hessians = INVERSE_HESSIANS if not args.quick else ["dense"]
    factors = SIZE_FACTORS if not args.quick else [0.4]

    records, trajectories = [], {}
    for name in names:
        problem = nonsmooth.PROBLEMS[name]
        print(f"\n{name} (n={n}, f(x0)={problem.f_x0(n):.4g}, f*={problem.target(n):.6g})")
        for direction in directions:
            for hessian in hessians:
                for factor in factors:
                    row, _ = run(problem, direction=direction,
                                 inverse_hessian=hessian, size_factor=factor,
                                 n=n, max_iterations=max_iterations)
                    records.append(row)
                    print(f"  {direction:<22} {hessian:<15} sf={factor:<5g} "
                          f"f={row['f final']:<14.7g} rel gap={row['rel gap']:>10.3e} "
                          f"calls={row['closure calls']:<6d} {row['status']}")
            # Trajectory for the figure: the best-scoring configuration of this
            # direction, replayed with recording on.
            same = [r for r in records
                    if r["problem"] == name and r["direction"] == direction]
            best = min(same, key=lambda r: abs(r["rel gap"]))
            _, trajectory = run(problem, direction=direction,
                                inverse_hessian=best["inverse_hessian"],
                                size_factor=best["size_factor"],
                                n=n, max_iterations=max_iterations, record=True)
            target = problem.target(n)
            for row in trajectory:
                row["gap"] = row["f"] - target
            trajectories[(name, direction)] = trajectory

    write_csv(records, "e0b_grid", EXPERIMENT)
    write_csv([{"problem": name, "direction": direction, **row}
               for (name, direction), rows in trajectories.items() for row in rows],
              "e0b_trajectories", EXPERIMENT)

    # Reference-matching layout: one row per problem, one column per direction,
    # holding the best result over the inverse-Hessian and size_factor axes.
    summary = []
    for name in names:
        problem = nonsmooth.PROBLEMS[name]
        row = {
            "problem": name,
            "convex": problem.is_convex,
            "f(x0)": problem.f_x0(n),
            "f* published": problem.target(n),
        }
        for direction, reference in DIRECTIONS.items():
            same = [r for r in records
                    if r["problem"] == name and r["direction"] == direction]
            if not same:
                continue
            best = min(same, key=lambda r: abs(r["rel gap"]))
            row[f"ours/{reference} f"] = best["f final"]
            row[f"ours/{reference} calls"] = best["closure calls"]
        # Reference solvers' own final values, when they have been transcribed.
        for solver, values in nonsmooth.REFERENCE_SOLVERS_N50.get(name, {}).items():
            row[f"published/{solver} f"] = values[0]
        summary.append(row)

    transcribed = bool(nonsmooth.REFERENCE_SOLVERS_N50)
    write_table(
        summary, "e0b_summary", EXPERIMENT,
        title=(f"E0b: best value reached per direction at n = {n}, against the "
               + ("published f* and the reference solvers' own final values."
                  if transcribed else
                  "published f*. The reference solvers' own final values "
                  "(Curtis & Que Tables 2-4) are NOT yet transcribed - see "
                  "REFERENCE_SOLVERS_N50 in paper/problems/nonsmooth.py.")),
    )
    if not transcribed:
        print("  note: reference-solver columns are empty "
              "(REFERENCE_SOLVERS_N50 not transcribed)")
    make_figure(trajectories, names)

    checks = Checks(enabled=args.check)
    if not args.quick and not args.problems and n == DIMENSION:
        register_predictions(checks, records)
    main_exit(checks)


if __name__ == "__main__":
    main()
