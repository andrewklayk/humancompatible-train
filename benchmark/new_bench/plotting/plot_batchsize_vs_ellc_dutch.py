"""plot_sweep.py — PBM penalty-range sweep and batch-size sweep, one curve per config.

2x2 grid, seed mean +- std over the ``opt`` split (full-batch values at each epoch end):
  columns: penalty_range at batch size --base-bs  |  batch size at penalty_range lower = --fixed-lo
  rows:    objective (loss)                      |  max_j c_j - bound   (<= 0 is feasible)

Needs runs that recorded their batch size (aggregate.py stores it as
hyperparameters["batch_size"]); older runs without it are ignored.

Usage (from benchmark/new_bench/, after scripts/dutch_pbm_sweep.sh):
    python aggregate.py --runs multirun/dutch_positive_rate_pair --out selection/dutch --approach opt
    python plotting/plot_sweep.py --agg selection/dutch/aggregated --penalty-update dimin --out plots/dutch_pbm_sweep_dimin.pdf
    python plotting/plot_sweep.py --agg selection/dutch/aggregated --penalty-update dimin_adapt --out plots/dutch_pbm_sweep_adapt.pdf
"""
import argparse
import os

import matplotlib.pyplot as plt

from prepare_results_plotting import ExperimentSpec, list_configs, config_params, metric_trajectory
from plot_style import set_neurips_style, TEXT_WIDTH, ROW_H


def _plot_column(axes, spec, method, where, label_of, sort_of, split):
    cfgs = sorted(list_configs(spec, method, where), key=lambda c: sort_of(config_params(spec, method, c)))
    if not cfgs:
        print(f"[warn] no {method} configs match {list(where)}")
    labels = [label_of(config_params(spec, method, c)) for c in cfgs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Several configs share a label in {labels}: they differ in other "
                         f"hyperparameters. Aggregate only this sweep's runs.")
    for c, label in zip(cfgs, labels):
        # aggregate.py stores max_viol as the raw max_j c_j, so subtract the bound here.
        for ax, metric, offset in ((axes[0], "loss", 0.0), (axes[1], "max_viol", 0.0)):
            tr = metric_trajectory(spec, method, c, split, metric)
            if tr is None:
                continue
            mean, std, _, epochs = tr
            line, = ax.plot(epochs, mean - offset, label=label)
            if std is not None:
                ax.fill_between(epochs, mean - offset - std, mean - offset + std,
                                color=line.get_color(), alpha=0.2, lw=0)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(cfgs),
                   handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    axes[1].axhline(0.1, color="k", lw=0.6, ls="--")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", default="selection/dutch/aggregated")
    ap.add_argument("--task", default="dutch_positive_rate_pair")
    ap.add_argument("--data", default="dutch")
    ap.add_argument("--method", default="pbm")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--base-bs", type=int, default=72, help="batch size of the penalty-range sweep")
    ap.add_argument("--fixed-lo", type=float, default=0.1, help="penalty_range lower bound of the batch-size sweep")
    ap.add_argument("--penalty-update", default="dimin", help="PBM penalty_update to plot (dimin | dimin_adapt)")
    ap.add_argument("--split", default="opt")
    ap.add_argument("--out", default="plots/dutch_pbm_sweep.pdf")
    args = ap.parse_args()

    set_neurips_style()
    spec = ExperimentSpec("sweep", args.task, args.data, args.bound, agg_root=args.agg)
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 2 * ROW_H), sharex=True, sharey="row")



    _plot_column(axes[:, 0], spec, args.method,
                 where={"batch_size": args.base_bs, "dual.penalty_update": args.penalty_update},
                 label_of=lambda p: rf"$\ell_c$ = {p['dual.penalty_range'][0]:g}",
                 sort_of=lambda p: p["dual.penalty_range"][0], split=args.split)
    _plot_column(axes[:, 1], spec, args.method,
                 where={"dual.penalty_range": lambda v: abs(v[0] - args.fixed_lo) < 1e-12,
                        "batch_size": lambda v: True,  # require a recorded batch size
                        "dual.penalty_update": args.penalty_update},
                 label_of=lambda p: rf"$B$ = {p['batch_size']}",
                 sort_of=lambda p: p["batch_size"], split=args.split)

    # axes[0, 0].set_title(rf"PBM like $\pi^k$ with varying $\ell_c$ ($B$ = {args.base_bs})", pad=16)
    # axes[0, 1].set_title(rf"PBM like $\pi^k$ with varying $B$ ($\ell_c$ = {args.fixed_lo:g})", pad=16)
    axes[0, 0].set_ylabel("Mean Loss")
    axes[1, 0].set_ylabel(r"$\max_j c_j$")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
