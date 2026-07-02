"""plot_cdf.py (new_bench) — Robustness CDF / performance profile.

For each method, plots the fraction of ALL its grid configs that are BOTH feasible
(seed-mean max constraint <= bound) AND have loss <= L, as a function of L.

  - Higher curve  = more configs reach that (loss, feasible) bar = more robust.
  - Plateau height = the method's overall feasibility rate.

Equivalent to ../../plotting/plot_cdf.py; only the data backend differs
(reads aggregate.py's per-cell aggregates -- run aggregate.py first).

Usage (run aggregate.py first):
    python plot_cdf.py --agg <aggregated_dir> --task <task> --data <data> --bound 0.1 \
        [--split train|val|test] [--tail K] [--out cdf.pdf]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from benchmark.new_bench.plotting.prepare_results_plotting import ExperimentSpec, aggregate_experiment
from plot_style import set_neurips_style, style_for, COL_WIDTH


def _print_frontier(agg, bound):
    print("\n--- feasible frontier (sorted by loss) ---")
    for method, df in agg.items():
        feas = df[df["feasible_mean_viol"]].sort_values("loss_mean")
        n_feas, n_tot = len(feas), len(df)
        best = f"{feas['loss_mean'].iloc[0]:.3f}" if n_feas else "—"
        print(f"  {method:10s}: {n_feas:3d}/{n_tot} feasible | best feasible loss {best} "
              f"| mean viol {df['violation_constr_mean'].mean():.4f} "
              f"| mean viol_std {df['violation_constr_std'].mean():.4f}")


def _cdf_curve(df, loss_grid):
    """fraction of ALL configs that are feasible AND loss_mean <= L, for each L."""
    n_total = len(df)
    feas = df[df["feasible_mean_viol"]]
    losses = np.sort(feas["loss_mean"].values)
    counts = np.searchsorted(losses, loss_grid, side="right")
    return counts / n_total


def plot_cdf(spec, methods=None, out="cdf.pdf", tail=5, split="train", hi=0.9):
    set_neurips_style()
    agg = aggregate_experiment(spec, tail=tail, split=split)
    if methods is None:
        methods = [m for m in ["adam", "pbm", "alm_proj", "alm_max", "ssg"] if m in agg]
    if not methods:
        print("no methods to plot")
        return None

    _print_frontier(agg, spec.bound)

    feas_losses = [agg[m][agg[m]["feasible_mean_viol"]]["loss_mean"].values
                   for m in methods if agg[m]["feasible_mean_viol"].any()]
    all_feas = np.concatenate(feas_losses) if feas_losses else np.array([0.0, 1.0])
    lo = all_feas.min()
    pad = 0.05 * (hi - lo + 1e-9)
    loss_grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.8))
    for m in methods:
        st = style_for(m)
        ax.plot(loss_grid, _cdf_curve(agg[m], loss_grid), label=st["label"],
                color=st["color"], ls=st["ls"], drawstyle="steps-post")
    ax.set_xlabel("Loss threshold $L$")
    ax.set_ylabel("Frac. configs feasible \\& $\\leq L$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="../selection/aggregated",
                    help="dir of aggregate.py's per-cell <cell>.csv/.json (run aggregate.py first)")
    ap.add_argument("--task", default="folktables_positive_rate_pair")
    ap.add_argument("--data", default="income")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--hi", type=float, default=0.9, help="upper x-limit (loss)")
    ap.add_argument("--out", default="plots/cdf.pdf")
    args = ap.parse_args()
    spec = ExperimentSpec(name=args.task, task=args.task, data=args.data,
                          bound=args.bound, agg_root=args.agg)
    plot_cdf(spec, out=args.out, tail=args.tail, split=args.split, hi=args.hi)
