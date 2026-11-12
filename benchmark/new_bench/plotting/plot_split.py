"""plot_split.py — PBM with shared vs. independent (split) loss/constraint samples.

Baseline: batch size --base-bs, loss and constraints on the same batch.
Split:    batch size 2*--base-bs, loss on one half and constraints on the other
          (scripts/dutch_split_batch.sh), run for twice the epochs.
Cgrad:    batch size 2*--base-bs, constraint value on one half, loss and constraint
          gradient on the other (MODE=cgrad scripts/dutch_split_batch.sh).
Cgradvar: cgrad + the covariance/variance term of shared sampling (MODE=cgradvar).
One curve per --curves entry (penalty_update:penalty_range_lo:shared|split|cgrad|cgradvar); the default
is dimin 0.001 shared vs. split, and the same pair for adapt.
The x-axis counts optimizer steps in baseline epochs (epoch * base_bs / B), so the
curves line up at an equal number of steps. Rows: loss | max_j c_j | max_j lambda_j
(opt split; lambda_j is the seed mean, its max over the constraints j is plotted, log axis).

    python plotting/plot_split.py --agg selection/dutch/aggregated --out plots/dutch_split_batch.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from prepare_results_plotting import (ExperimentSpec, list_configs, config_params, metric_trajectory,
                                     dual_trajectory)
from plot_style import set_neurips_style, TEXT_WIDTH, ROW_H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", default="selection/dutch/aggregated")
    ap.add_argument("--task", default="dutch_positive_rate_pair")
    ap.add_argument("--data", default="dutch")
    ap.add_argument("--method", default="pbm")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--base-bs", type=int, default=72)
    ap.add_argument("--curves", nargs="+",
                    default=["dimin:0.001:shared", "dimin:0.001:split",
                             "dimin_adapt:0.001:shared", "dimin_adapt:0.001:split"],
                    help="curves as penalty_update:penalty_range_lo:shared|split|cgrad|cgradvar")
    ap.add_argument("--split", default="opt")
    ap.add_argument("--out", default="plots/dutch_split_batch.pdf")
    args = ap.parse_args()

    set_neurips_style()
    spec = ExperimentSpec("split", args.task, args.data, args.bound, agg_root=args.agg)
    curves = []  # (label, config index, batch size)
    for item in args.curves:
        update, lo, mode = item.split(":")
        if mode not in ("shared", "split", "cgrad", "cgradvar"):
            raise SystemExit(f"{item}: mode must be shared, split, cgrad or cgradvar")
        bs = args.base_bs if mode == "shared" else 2 * args.base_bs
        where = {"dual.penalty_range": lambda v, lo=float(lo): abs(v[0] - lo) < 1e-12,
                 "dual.penalty_update": update, "batch_size": bs}
        # shared runs have neither key; split runs split_batch=True; cgrad runs split_cgrad=True
        def flags(c):
            p = config_params(spec, args.method, c)
            return tuple(bool(p.get(k, False)) for k in ("split_batch", "split_cgrad", "cgrad_var"))
        want = {"shared": (False, False, False), "split": (True, False, False),
                "cgrad": (False, True, False), "cgradvar": (False, True, True)}[mode]
        match = [c for c in list_configs(spec, args.method, where) if flags(c) == want]
        if len(match) != 1:
            raise SystemExit(f"{item}: expected 1 config, found {len(match)} (not run / not aggregated?)")
        name = r"$\hat{\pi}^{\operatorname{ADAPT}}$" if update == "dimin_adapt" else r"$\pi^k$"
        mode_name = {"shared": "Shared Samples", "split": "Independent Samples",
                     "cgrad": r"Independent $c$ / $\nabla c$",
                     "cgradvar": r"Independent $c$ / $\nabla c$ + Var"}[mode]
        curves.append((rf"{name}, $\ell_c$={float(lo):g}, {mode_name}", match[0], bs))

    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH / 2, 3 * ROW_H), sharex=True)
    loss_floor = np.inf  # lowest mean loss over all curves: bottom of the log axis
    for label, c, bs in curves:
        for ax, metric in zip(axes, ("loss", "max_viol")):
            tr = metric_trajectory(spec, args.method, c, args.split, metric)
            if tr is None:
                continue
            mean, std, _, epochs = tr
            x = epochs * args.base_bs / bs  # steps/epoch scale as 1/B
            line, = ax.plot(x, mean, label=label)
            lower = None if std is None else mean - std
            if metric == "loss":
                loss_floor = min(loss_floor, np.nanmin(mean))
                if lower is not None:
                    # diverged runs have std > mean; keep the band on the positive log axis
                    lower = np.maximum(lower, 0.5 * mean)
            if std is not None:
                ax.fill_between(x, lower, mean + std, color=line.get_color(), alpha=0.2, lw=0)
        # duals: seed-mean lambda_j per constraint, max over j (no band: a max of means)
        dt = dual_trajectory(spec, args.method, c, args.split)
        if dt is not None:
            lam_mean, _, epochs = dt
            axes[2].plot(epochs * args.base_bs / bs, lam_mean.max(axis=0), label=label)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize="small",
                   handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    axes[1].axhline(args.bound, color="k", lw=0.6, ls="--")
    # axes[0].set_yscale("log")  # diverged cgrad runs reach ~1e3, stable ones sit near 0.5
    if np.isfinite(loss_floor):
        axes[0].set_ylim(bottom=0.9 * loss_floor)
    axes[0].set_ylabel("Mean Loss")
    axes[1].set_ylabel(r"$\max_j c_j$")
    axes[2].set_yscale("log")  # duals span ~1e-1 (stable) to the 1e2 cap (diverged cgrad)
    axes[2].set_ylabel(r"$\max_j \lambda_j$")
    axes[2].set_xlabel(rf"Steps (in epochs at $B$ = {args.base_bs})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
