"""plot_split.py — PBM with shared vs. independent (split) loss/constraint samples.

Baseline: batch size --base-bs, loss and constraints on the same batch.
Split:    batch size 2*--base-bs, loss on one half and constraints on the other
          (scripts/dutch_split_batch.sh), run for twice the epochs.
The x-axis counts optimizer steps in baseline epochs (epoch * base_bs / B), so the
two curves line up at an equal number of steps. Rows: loss | max_j c_j (opt split).

    python plotting/plot_split.py --agg selection/dutch/aggregated --out plots/dutch_split_batch.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt

from prepare_results_plotting import ExperimentSpec, list_configs, config_params, metric_trajectory
from plot_style import set_neurips_style, TEXT_WIDTH, ROW_H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", default="selection/dutch/aggregated")
    ap.add_argument("--task", default="dutch_positive_rate_pair")
    ap.add_argument("--data", default="dutch")
    ap.add_argument("--method", default="pbm")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--base-bs", type=int, default=72)
    ap.add_argument("--range-lo", type=float, default=0.001, help="penalty_range lower bound")
    ap.add_argument("--penalty-update", default="dimin")
    ap.add_argument("--split", default="opt")
    ap.add_argument("--out", default="plots/dutch_split_batch.pdf")
    args = ap.parse_args()

    set_neurips_style()
    spec = ExperimentSpec("split", args.task, args.data, args.bound, agg_root=args.agg)
    where = {"dual.penalty_range": lambda v: abs(v[0] - args.range_lo) < 1e-12,
             "dual.penalty_update": args.penalty_update,
             "batch_size": lambda v: v in (args.base_bs, 2 * args.base_bs)}
    curves = {}
    for c in list_configs(spec, args.method, where):
        p = config_params(spec, args.method, c)
        split = bool(p.get("split_batch", False))
        # baseline = shared batch at base_bs; split = split batch at 2*base_bs
        if p["batch_size"] == (2 * args.base_bs if split else args.base_bs):
            curves[split] = (c, p["batch_size"])
    if set(curves) != {False, True}:
        raise SystemExit(f"need one shared and one split config, found split={sorted(curves)}")

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH / 2, 2 * ROW_H), sharex=True)
    for split, (c, bs) in sorted(curves.items()):
        label = (rf"independent ($2\times${args.base_bs})" if split else rf"shared ($B$ = {bs})")
        for ax, metric in zip(axes, ("loss", "max_viol")):
            tr = metric_trajectory(spec, args.method, c, args.split, metric)
            if tr is None:
                continue
            mean, std, _, epochs = tr
            x = epochs * args.base_bs / bs  # steps/epoch scale as 1/B
            line, = ax.plot(x, mean, label=label)
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, lw=0)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                   handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    axes[1].axhline(args.bound, color="k", lw=0.6, ls="--")
    axes[0].set_ylabel("Mean Loss")
    axes[1].set_ylabel(r"$\max_j c_j$")
    axes[1].set_xlabel(rf"Steps (in epochs at $B$ = {args.base_bs})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
