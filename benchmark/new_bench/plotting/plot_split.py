"""plot_split.py — PBM with shared vs. independent (split) loss/constraint samples.

Baseline: batch size --base-bs, loss and constraints on the same batch.
Split:    batch size 2*--base-bs, loss on one half and constraints on the other
          (scripts/dutch_split_batch.sh), run for twice the epochs.
One curve per --curves entry (penalty_update:penalty_range_lo:shared|split); the default
is dimin 0.001 shared vs. split, and the same pair for adapt.
The x-axis counts optimizer steps in baseline epochs (epoch * base_bs / B), so the
curves line up at an equal number of steps. Rows: loss | max_j c_j (opt split).

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
    ap.add_argument("--curves", nargs="+",
                    default=["dimin:0.001:shared", "dimin:0.001:split",
                             "dimin_adapt:0.001:shared", "dimin_adapt:0.001:split"],
                    help="curves as penalty_update:penalty_range_lo:shared|split")
    ap.add_argument("--split", default="opt")
    ap.add_argument("--out", default="plots/dutch_split_batch.pdf")
    args = ap.parse_args()

    set_neurips_style()
    spec = ExperimentSpec("split", args.task, args.data, args.bound, agg_root=args.agg)
    curves = []  # (label, config index, batch size)
    for item in args.curves:
        update, lo, mode = item.split(":")
        split, bs = mode == "split", (2 * args.base_bs if mode == "split" else args.base_bs)
        where = {"dual.penalty_range": lambda v, lo=float(lo): abs(v[0] - lo) < 1e-12,
                 "dual.penalty_update": update, "batch_size": bs}
        # shared runs have no split_batch key; split runs have split_batch=True
        match = [c for c in list_configs(spec, args.method, where)
                 if bool(config_params(spec, args.method, c).get("split_batch", False)) == split]
        if len(match) != 1:
            raise SystemExit(f"{item}: expected 1 config, found {len(match)} (not run / not aggregated?)")
        name = r"$\hat{\pi}^{\operatorname{ADAPT}}$" if update == "dimin_adapt" else r"$\pi^k$"
        mode_name = "Shared Samples" if mode == "shared" else "Independent Samples"
        curves.append((rf"{name}, $\ell_c$={float(lo):g}, {mode_name}", match[0], bs))

    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH / 2, 2 * ROW_H), sharex=True)
    for label, c, bs in curves:
        for ax, metric in zip(axes, ("loss", "max_viol")):
            tr = metric_trajectory(spec, args.method, c, args.split, metric)
            if tr is None:
                continue
            mean, std, _, epochs = tr
            x = epochs * args.base_bs / bs  # steps/epoch scale as 1/B
            line, = ax.plot(x, mean, label=label)
            if std is not None:
                ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2, lw=0)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize="small",
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
