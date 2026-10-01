"""
E2c -- batch size and constraint-update cadence.

Two panels, sharing ``a_fairness.train()``/``METHODS``/``fairness.build()``:

**Batch size** (cadence fixed at per-iteration): ``per_group`` in {4, 8, 16}, via
``fairness.build(..., per_group=..., extend_groups=True)``. ``extend_groups=True``
oversamples smaller groups so the epoch runs as long as the *largest* group
allows -- without it, steps/epoch shrinks with ``per_group`` and confounds "less
noise" with "fewer dual updates per epoch". Oversampling skews the *objective*'s
implicit group weighting (not the constraint -- every batch is already
group-balanced either way), so every run here also passes ``reweight_loss=True``,
correcting the training loss with ``BalancedBatchSampler.group_weights``.

**Cadence** (``per_group`` fixed at 8): per-iteration against per-epoch (frozen
duals through the epoch, one dual update at the end from a full-batch evaluation
on ``problem.train`` -- the classical outer/inner ALM structure).

Per-cell hyperparameters (primal_lr, dual_lr) are tuned outside this script, via
``paper/tune.py``'s Hydra multirun, then applied here from a small committed file:

    python paper/tune.py -m experiment=e2c_per_iteration +sweep=e2c_alm \\
        experiment.problem.dataset=income,dutch experiment.problem.per_group=4,8,16
    python paper/tune.py -m experiment=e2c_per_epoch +sweep=e2c_alm \\
        experiment.problem.dataset=income,dutch
    python paper/e2/tune_apply.py --runs paper/multirun/e2c/alm

    python paper/e2/c_cadence.py --quick      # one problem, one cell, 2 epochs, no tuning
    python paper/e2/c_cadence.py --check      # full run, non-zero exit on a failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml

from humancompatible.train.dual_optim import ALM
from paper._harness import Checks, figure, main_exit, save_figure, write_csv, write_table
from paper.e2 import a_fairness
from paper.problems import fairness

EXPERIMENT = "e2c"

EPOCHS = a_fairness.EPOCHS
SEEDS = a_fairness.SEEDS
TAIL = a_fairness.TAIL

# (dataset, shape, bound) -- the two "pairwise" fairness problems, where per-group
# sample counts and the restart pathology are the live mechanism. The "agg"
# shape's aggregate norm constraint is far less sensitive to any one group's
# estimate noise, so it is left out to bound this script's runtime.
PROBLEMS = [("income", "pairwise", 0.05), ("dutch", "pairwise", 0.05)]

# (per_group, cadence). Two separate 1D sweeps sharing the (8, per_iteration)
# baseline cell -- not a full cross, to bound runtime.
CELLS = [(4, "per_iteration"), (8, "per_iteration"), (16, "per_iteration"),
         (8, "per_epoch")]
BATCH_GROUPS = [pg for pg, cadence in CELLS if cadence == "per_iteration"]

METHODS = a_fairness.METHODS
CONSTRAINED = a_fairness.CONSTRAINED
# Plain Adam never evaluates the constraint, so timing a dual method against it
# measures the fairret statistic, not the dual layer. This control pays the same
# constraint forward/backward as every dual method, via a zero-weighted term,
# isolating the dual *update*'s cost -- reused from a_fairness.py, which keeps it
# commented out in its own METHODS.
CONTROL = "Adam (c in graph)"
# ALM's structural variants -- everything METHODS carries beyond plain Adam.
ALM_FAMILY_KWARGS = {
    "ALM": {},
    "ALM (HPR)": {"augmentation": "hpr"},
    "ALM (restart)": {"restart": True},
}

# Where paper/e2/tune_apply.py writes the winner of each (problem, per_group,
# cadence) cell's paper/tune.py sweep -- see the module docstring's Usage.
TUNED_PATH = Path(__file__).resolve().parents[1] / "conf" / "e2c_tuned.yaml"


def train(problem, seed, epochs, *, primal_lr, dual_lr, cadence):
    """One ALM run at a given (primal_lr, dual_lr, cadence).

    The unit ``paper/tune.py``'s ``e2c`` adapter calls once per hyperparameter
    combination -- everything else in this script's tuning story (the grid, the
    multirun, picking a winner) lives outside it, in the Hydra configs and
    ``tune_apply.py``.
    """
    factory = lambda m: ALM(m=m, lr=dual_lr, penalty=1.0, is_ineq=True)
    return a_fairness.train(
        problem, "ALM", seed, epochs, dual_factory=factory,
        primal_lr=primal_lr, cadence=cadence, reweight_loss=True,
    )


def _load_tuned():
    if not TUNED_PATH.exists():
        return {}
    return yaml.safe_load(TUNED_PATH.read_text()) or {}


_TUNED = _load_tuned()


def _tuned(problem_name, per_group, cadence):
    """(primal_lr, dual_lr) for one cell, from TUNED_PATH.

    Raises rather than silently falling back to an untuned value: a full run's
    whole point is per-cell-tuned hyperparameters (see the module docstring's
    "Hyperparameters are tuned per cell" paragraph in paper/README.md), so a
    missing cell is a setup error, not a value to guess at.
    """
    key = f"{problem_name}_{per_group}_{cadence}"
    cell = _TUNED.get(key)
    if cell is None:
        raise SystemExit(
            f"{key!r} not in {TUNED_PATH} -- run the tune.py sweep and "
            f"tune_apply.py first (see this script's module docstring)."
        )
    return float(cell["primal_lr"]), float(cell["dual_lr"])


def _dual_factory(method, dual_lr):
    kwargs = ALM_FAMILY_KWARGS[method]
    return lambda m: ALM(m=m, lr=dual_lr, penalty=1.0, is_ineq=True, **kwargs)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="E2c: batch size and constraint-update cadence"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--from-csv", action="store_true",
        help="re-evaluate the predictions against the committed results instead of "
             "retraining.",
    )
    args = parser.parse_args(argv)

    if args.from_csv:
        rows, trajectories = _reload()
        _finish(rows, trajectories, args)
        return

    epochs = 2 if args.quick else args.epochs
    seeds = SEEDS[:1] if args.quick else SEEDS
    specs = PROBLEMS[:1] if args.quick else PROBLEMS
    cells = CELLS[:1] if args.quick else CELLS

    rows, trajectories = [], []
    for dataset, shape, bound in specs:
        for per_group, cadence in cells:
            problem = fairness.build(dataset, shape, bound=bound, per_group=per_group,
                                      extend_groups=True)
            print(f"\n=== {problem.name}: per_group={per_group}, cadence={cadence}, "
                  f"batch {problem.batch_size}, {len(problem.loader)} batches/epoch ===")

            if args.quick:
                primal_lr, dual_lr = a_fairness.PRIMAL_LR, a_fairness.DUAL_LR
            else:
                primal_lr, dual_lr = _tuned(problem.name, per_group, cadence)
            print(f"    tuned ALM: primal_lr={primal_lr}, dual_lr={dual_lr}")

            # CONTROL only at per_iteration: it is cadence-invariant (no duals), and
            # B3 -- the only prediction that needs it -- only looks at per_iteration
            # cells. It pays the constraint's forward and backward like every dual
            # method does, unlike plain Adam, so it is what isolates the dual
            # *update*'s cost instead of also timing the constraint evaluation.
            cell_methods = list(METHODS) + ([CONTROL] if cadence == "per_iteration" else [])
            for method in cell_methods:
                if method == "Adam" and cadence != "per_iteration":
                    # Cadence is a no-op without duals; per_iteration already covers it.
                    continue
                if method in ("Adam", CONTROL):
                    factory = (lambda m: None) if method == CONTROL else None
                    run_primal_lr = a_fairness.PRIMAL_LR
                else:
                    factory = _dual_factory(method, dual_lr)
                    run_primal_lr = primal_lr
                for seed in seeds:
                    history = a_fairness.train(
                        problem, method, seed, epochs, dual_factory=factory,
                        primal_lr=run_primal_lr, cadence=cadence, reweight_loss=True,
                    )
                    for row in history:
                        trajectories.append({
                            "problem": problem.name, "per_group": per_group,
                            "cadence": cadence, "method": method, "seed": seed, **row,
                        })
                    tail = history[-min(TAIL, len(history) - 1):]

                    def mean(key, source=tail):
                        return float(np.mean([h[key] for h in source]))

                    rows.append({
                        "problem": problem.name, "per_group": per_group,
                        "cadence": cadence, "method": method, "seed": seed,
                        "steps/epoch": len(problem.loader),
                        "train loss": mean("train_loss"),
                        "test loss": mean("test_loss"),
                        "train max_viol": mean("train_max_viol"),
                        "test max_viol": mean("test_max_viol"),
                        "train max_viol osc": float(np.std(
                            [h["train_max_viol"] for h in tail])),
                        "s/epoch": mean("epoch_time", history[1:]),
                    })
                    row = rows[-1]
                    print(f"  {method:<17} seed={seed}  viol train "
                          f"{row['train max_viol']:+.4f}  osc "
                          f"{row['train max_viol osc']:.4f}  "
                          f"({row['s/epoch']:.2f} s/epoch)")

    write_csv(trajectories, "e2c_trajectories", EXPERIMENT)
    write_csv(rows, "e2c_final", EXPERIMENT)
    _finish(rows, trajectories, args)


def _finish(rows, trajectories, args):
    summary = _summarize(rows)
    write_table(summary, "e2c_summary", EXPERIMENT,
                title="E2c: tail-averaged end-of-training values, mean over seeds "
                      "(per-cell tuned ALM hyperparameters)")
    make_figures(summary, rows)

    checks = Checks(enabled=args.check)
    register_predictions(checks, summary, rows)
    main_exit(checks, EXPERIMENT, "e2c_predictions")


def _reload():
    import csv

    from paper._harness import RESULTS

    def read(name):
        path = RESULTS / EXPERIMENT / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run without --from-csv first")
        out = []
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                out.append({key: _coerce(value) for key, value in record.items()})
        return out

    rows, trajectories = read("e2c_final"), read("e2c_trajectories")
    print(f"reloaded {len(rows)} final rows, {len(trajectories)} trajectory rows")
    return rows, trajectories


def _coerce(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _summarize(rows):
    """Mean over seeds, one row per (problem, per_group, cadence, method)."""
    keys = ["train loss", "test loss", "train max_viol", "test max_viol",
            "train max_viol osc", "s/epoch", "steps/epoch"]
    cells = dict.fromkeys((r["problem"], r["per_group"], r["cadence"]) for r in rows)
    methods = dict.fromkeys(r["method"] for r in rows)  # includes CONTROL
    summary = []
    for problem, per_group, cadence in cells:
        for method in methods:
            group = [r for r in rows if r["problem"] == problem
                     and r["per_group"] == per_group and r["cadence"] == cadence
                     and r["method"] == method]
            if not group:
                continue
            entry = {"problem": problem, "per_group": per_group, "cadence": cadence,
                     "method": method, "seeds": len(group)}
            for key in keys:
                entry[key] = float(np.mean([r[key] for r in group]))
            summary.append(entry)
    return summary


def _entry(summary, problem, per_group, cadence, method):
    for row in summary:
        if (row["problem"] == problem and row["per_group"] == per_group
                and row["cadence"] == cadence and row["method"] == method):
            return row
    return None


def _seed_values(rows, key, *, problem, per_group, cadence, method):
    return [r[key] for r in rows if r["problem"] == problem
            and r["per_group"] == per_group and r["cadence"] == cadence
            and r["method"] == method]


def _restart_gaps(rows, problem, per_group, cadence):
    """Per-seed (ALM(restart) - ALM) train max_viol, paired by seed."""
    restart = _seed_values(rows, "train max_viol", problem=problem, per_group=per_group,
                           cadence=cadence, method="ALM (restart)")
    alm = _seed_values(rows, "train max_viol", problem=problem, per_group=per_group,
                       cadence=cadence, method="ALM")
    return [r - a for r, a in zip(restart, alm)]


def _resolved(a, b):
    """mean(a) - mean(b), gated on the 2-SE-over-samples interval excluding zero.
    Returns (diff, resolved, detail); resolved is forced False
    on fewer than 2 samples per side (e.g. a --quick run missing a cell) instead
    of computing a degenerate standard error."""
    if len(a) < 2 or len(b) < 2:
        return 0.0, False, f"insufficient data ({len(a)} vs {len(b)} samples)"
    diff = float(np.mean(a) - np.mean(b))
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    resolved = abs(diff) > 2 * se
    detail = f"{np.mean(a):+.4f} vs {np.mean(b):+.4f}  (diff {diff:+.4f} +/- {se:.4f} SE)"
    return diff, resolved, detail


# --------------------------------------------------------------------------- #
# predictions
# --------------------------------------------------------------------------- #


def register_predictions(checks, summary, rows):
    problems = list(dict.fromkeys(r["problem"] for r in summary))
    small, large = BATCH_GROUPS[0], BATCH_GROUPS[-1]

    for name in problems:
        # B1
        osc_small = _seed_values(rows, "train max_viol osc", problem=name,
                                 per_group=small, cadence="per_iteration", method="ALM")
        osc_large = _seed_values(rows, "train max_viol osc", problem=name,
                                 per_group=large, cadence="per_iteration", method="ALM")
        diff, resolved, detail = _resolved(osc_small, osc_large)
        if not resolved:
            print(f"  note: B1 not resolvable on {name} -- {detail}")
        else:
            checks.expect(
                diff > 0,
                f"B1: ALM's train-violation oscillation is lower at per_group="
                f"{large} than at per_group={small} on {name}",
                f"osc(per_group={small}) vs osc(per_group={large}): {detail}",
            )

        # B2
        gaps_small = _restart_gaps(rows, name, small, "per_iteration")
        gaps_large = _restart_gaps(rows, name, large, "per_iteration")
        diff, resolved, detail = _resolved(gaps_small, gaps_large)
        if not resolved:
            print(f"  note: B2 not resolvable on {name} -- {detail}")
        else:
            checks.expect(
                diff > 0,
                f"B2: ALM(restart)'s feasibility gap to plain ALM is smaller at "
                f"per_group={large} than at per_group={small} on {name}",
                f"gap(per_group={small}) vs gap(per_group={large}): {detail}",
            )

        # B3 -- against CONTROL, not plain Adam: plain Adam never evaluates the
        # constraint, so timing against it would measure the fairret statistic
        # (a property of the problem) on top of the dual update itself.
        for per_group in BATCH_GROUPS:
            alm = _entry(summary, name, per_group, "per_iteration", "ALM")
            control = _entry(summary, name, per_group, "per_iteration", CONTROL)
            if alm is None or control is None:
                print(f"  note: B3 not resolvable on {name} at per_group={per_group} "
                      f"-- cell not run")
                continue
            per_step_us = (alm["s/epoch"] - control["s/epoch"]) / alm["steps/epoch"] * 1e6
            checks.expect(
                per_step_us < 500.0,
                f"B3: the per-step dual cost stays under 500 us on {name} at "
                f"per_group={per_group}",
                f"{per_step_us:.0f} us/step over {alm['steps/epoch']:.0f} steps",
            )

        # C1
        for method in CONSTRAINED:
            pi = _seed_values(rows, "train max_viol osc", problem=name, per_group=8,
                              cadence="per_iteration", method=method)
            pe = _seed_values(rows, "train max_viol osc", problem=name, per_group=8,
                              cadence="per_epoch", method=method)
            diff, resolved, detail = _resolved(pi, pe)
            if not resolved:
                print(f"  note: C1 not resolvable for {method} on {name} -- {detail}")
                continue
            checks.expect(
                diff > 0,
                f"C1: {method} has lower train-violation oscillation under "
                f"per-epoch cadence than per-iteration on {name}",
                f"per-iteration vs per-epoch osc: {detail}",
            )

        # C2
        gaps_pi = _restart_gaps(rows, name, 8, "per_iteration")
        gaps_pe = _restart_gaps(rows, name, 8, "per_epoch")
        diff, resolved, detail = _resolved(
            [abs(g) for g in gaps_pi], [abs(g) for g in gaps_pe]
        )
        if not resolved:
            print(f"  note: C2 not resolvable on {name} -- {detail}")
        else:
            checks.expect(
                diff > 0,
                f"C2: ALM(restart)'s |feasibility gap| to plain ALM is smaller "
                f"under per-epoch cadence than per-iteration at per_group=8 on "
                f"{name}",
                f"|gap| per-iteration vs per-epoch: {detail}",
            )

        # C3 -- same method (ALM), both cadences: this cancels the shared per-step
        # constraint-evaluation cost that both pay identically, isolating exactly
        # what switching cadence adds -- one full-batch forward+constraint pass at
        # epoch end. A relative bound (half of per-iteration's own per-epoch cost)
        # rather than a hand-picked absolute one, since that one pass's cost scales
        # with the dataset, not with anything this script controls.
        alm_pi = _entry(summary, name, 8, "per_iteration", "ALM")
        alm_pe = _entry(summary, name, 8, "per_epoch", "ALM")
        if alm_pi is None or alm_pe is None:
            print(f"  note: C3 not resolvable on {name} -- per_epoch cell not run")
            continue
        added = alm_pe["s/epoch"] - alm_pi["s/epoch"]
        checks.expect(
            added < 0.5 * alm_pi["s/epoch"],
            f"C3: per-epoch cadence's one extra full-batch update per epoch adds "
            f"less than half of per-iteration's own per-epoch cost on {name} -- "
            f"batching the dual update does not double the epoch",
            f"+{added * 1e3:.1f} ms/epoch added, {added / alm_pi['s/epoch'] * 100:.0f}% "
            f"of per-iteration's {alm_pi['s/epoch']:.2f} s/epoch",
        )


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def make_figures(summary, rows):
    problems = list(dict.fromkeys(r["problem"] for r in summary))

    # Batch-size panel: oscillation vs. per_group, one line per method.
    fig, axes, plt = figure(1, len(problems), row_height=2.4)
    for ax, name in zip(axes, problems):
        for method in CONSTRAINED:
            values = [_entry(summary, name, pg, "per_iteration", method) for pg in BATCH_GROUPS]
            if any(v is None for v in values):
                continue
            ax.plot(BATCH_GROUPS, [v["train max_viol osc"] for v in values],
                    marker="o", markersize=4, label=method)
        ax.set_xlabel("per_group")
        ax.set_xticks(BATCH_GROUPS)
        ax.set_title(name)
    axes[0].set_ylabel("train max-violation\noscillation (tail std)")
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:  # e.g. a --quick run that only covers one per_group point
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "e2c_batch_size", EXPERIMENT)
    plt.close(fig)

    # Cadence panel: paired bars, per-iteration vs per-epoch, at per_group=8.
    fig, axes, plt = figure(1, len(problems), row_height=2.4)
    width = 0.35
    x = np.arange(len(CONSTRAINED))
    for ax, name in zip(axes, problems):
        for offset, cadence in ((-width / 2, "per_iteration"), (width / 2, "per_epoch")):
            values = [_entry(summary, name, 8, cadence, method) for method in CONSTRAINED]
            heights = [v["train max_viol osc"] if v else 0.0 for v in values]
            ax.bar(x + offset, heights, width, label=cadence)
        ax.set_xticks(x)
        ax.set_xticklabels(CONSTRAINED, rotation=20, ha="right")
        ax.set_title(name)
    axes[0].set_ylabel("train max-violation\noscillation (tail std)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_figure(fig, "e2c_cadence", EXPERIMENT)
    plt.close(fig)


if __name__ == "__main__":
    main()
