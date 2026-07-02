"""plot_pareto.py (new_bench) — Loss vs. constraint-violation scatter.

Each point = one hyperparameter config of one algorithm:
    x = max constraint violation (minus bound), mean ± std over seeds
    y = loss, mean ± std over seeds
A vertical dotted line marks the feasibility bound (x = 0).

Equivalent to ../../plotting/plot_pareto.py; only the data backend differs
(reads aggregate.py's per-cell aggregates -- run aggregate.py first).

Usage (run aggregate.py first):
    python plot_pareto.py --agg <aggregated_dir> --task <task> --data <data> --bound 0.1 \
        [--split train|val|test] [--tail K] [--pareto] [--out pareto.pdf]
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from benchmark.new_bench.plotting.prepare_results_plotting import ExperimentSpec, aggregate_experiment
from plot_style import set_neurips_style, style_for, COL_WIDTH

METHOD_MARKER = {"adam": "o", "pbm": "s", "alm_proj": "^", "alm_max": "D", "ssg": "v"}
_FALLBACK_MARKERS = ["P", "X", "h", "*", "p"]


def _pareto_mask(xs, ys):
    """Boolean mask of non-dominated points (minimise both x and y)."""
    mask = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                mask[i] = False
                break
    return mask


def plot_scatter(spec, methods=None, out="pareto.pdf", tail=1, split="train", pareto_filter=False):
    set_neurips_style()
    agg = aggregate_experiment(spec, tail=tail, split=split)
    if methods is None:
        methods = [m for m in ["pbm", "alm_proj", "alm_max", "ssg"] if m in agg]
    if not methods:
        print("no methods to plot")
        return None

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.85))
    fb = 0
    for m in methods:
        df = agg[m]
        st = style_for(m)
        marker = METHOD_MARKER.get(m, _FALLBACK_MARKERS[fb % len(_FALLBACK_MARKERS)])
        if m not in METHOD_MARKER:
            fb += 1

        ys, y_err = df["loss_mean"].values, df["loss_std"].values
        xs = df["violation_constr_mean"].values - spec.bound
        x_err = df["violation_constr_std"].values

        if pareto_filter:
            mask = _pareto_mask(xs, ys)
            xs, ys, y_err, x_err = xs[mask], ys[mask], y_err[mask], x_err[mask]

        ax.errorbar(xs, ys, xerr=x_err, yerr=y_err, fmt=marker, color=st["color"],
                    label=st["label"], ms=4, lw=0.5, capsize=2, capthick=0.5,
                    zorder=3, alpha=0.85)

    ax.axvline(0, color="#888888", lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("Mean max constraint violation (± std over seeds)")
    ax.set_ylabel(f"{split.capitalize()} loss (mean ± std over seeds)")
    fig.legend(loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    ap.add_argument("--tail", type=int, default=1)
    ap.add_argument("--pareto", action="store_true", help="keep only non-dominated configs per method")
    ap.add_argument("--out", default="plots/pareto.pdf")
    args = ap.parse_args()
    spec = ExperimentSpec(name=args.task, task=args.task, data=args.data,
                          bound=args.bound, agg_root=args.agg)
    plot_scatter(spec, out=args.out, tail=args.tail, split=args.split, pareto_filter=args.pareto)
