"""plot_kkt.py (new_bench) — closeness-to-KKT over configurations (opt approach).

Reads the seed-averaged ``opt`` split that ``aggregate.py`` persists for
``approach=opt`` runs (full-batch KKT metrics: ``grad_norm`` = ‖∇L‖ stationarity,
``max_viol`` = primal feasibility, ``compl`` = complementarity). The composite KKT
residual per config/epoch is

    r = ‖∇L‖ + max(0, max_viol) + |compl|

(``max_viol``/``compl`` are absent for unconstrained/SSG methods -> those terms are 0).

Each config is collapsed to a final scalar = mean of its last ``--tail`` epochs.

Modes (``--mode``):
  cdf      (default) per method, fraction of configs with final residual <= r vs r
  pdf      per method, histogram of final residuals (the PDF of the CDF)
  scatter  one point per config, grouped by method (final residual, log y)
  conv     residual vs epoch: faint line per config + bold best-config per method
  all      cdf + pdf + scatter + conv in a 2x2 grid

``--metric {residual,grad_norm,max_viol,compl}`` plots a single component instead.

Run ``aggregate.py --approach opt --out selection/opt`` first, then point
``--agg`` at ``selection/opt/aggregated``.

Usage:
    python plot_kkt.py --agg ../selection/opt/aggregated \
        --task folktables_positive_rate_pair --data income \
        --mode cdf --metric residual --tail 5 --out plots/kkt_cdf.pdf
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from prepare_results_plotting import ExperimentSpec, list_configs, metric_trajectory
from plot_style import set_neurips_style, style_for, COL_WIDTH

SPLIT = "opt"
METHODS = ["adam", "pbm", "pbm_mirror", "alm_proj", "alm_max", "ssg"]
_AXLABEL = {"residual": "KKT residual $r$", "grad_norm": r"$\|\nabla_x L\|$",
            "max_viol": "feasibility $\\max_j(c_j-b)_+$", "compl": "complementarity $\\sum_j|\\lambda_j g_j|$"}


def _residual_traj(spec, method, cfg):
    """(r[L], epochs[L]) with r = ‖∇L‖ + relu(max_viol) + |compl|, or None."""
    gn = metric_trajectory(spec, method, cfg, SPLIT, "grad_norm")
    if gn is None:
        return None
    r = np.asarray(gn[0], dtype=float)
    mv = metric_trajectory(spec, method, cfg, SPLIT, "max_viol")
    if mv is not None:
        r = r + np.clip(np.asarray(mv[0], dtype=float), 0, None)
    cp = metric_trajectory(spec, method, cfg, SPLIT, "compl")
    if cp is not None:
        r = r + np.abs(np.asarray(cp[0], dtype=float))
    return r, gn[3]


def _metric_traj(spec, method, cfg, metric):
    if metric == "residual":
        return _residual_traj(spec, method, cfg)
    t = metric_trajectory(spec, method, cfg, SPLIT, metric)
    if t is None:
        return None
    vals = np.asarray(t[0], dtype=float)
    if metric == "max_viol":
        vals = np.clip(vals, 0, None)  # plot the violation (feasibility) side only
    return vals, t[3]


def _collect_final(spec, method, metric, tail):
    """[(cfg, final_scalar, traj, epochs)] over all configs of a method (skips missing)."""
    out = []
    for cfg in list_configs(spec, method):
        t = _metric_traj(spec, method, cfg, metric)
        if t is None or len(t[0]) == 0:
            continue
        out.append((cfg, float(np.mean(t[0][-tail:])), t[0], t[1]))
    return out


def _finals_by_method(spec, methods, metric, tail):
    d = {}
    for m in methods:
        rows = _collect_final(spec, m, metric, tail)
        if rows:
            d[m] = rows
    return d


def _plot_cdf(ax, finals, log):
    for m, rows in finals.items():
        st = style_for(m)
        vals = np.sort([r[1] for r in rows])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=st["label"], color=st["color"], ls=st["ls"],
                drawstyle="steps-post")
    ax.set_ylabel("frac. configs $\\leq r$")
    ax.set_ylim(-0.02, 1.02)
    if log:
        ax.set_xscale("log")
    ax.legend(loc="lower right")


def _plot_pdf(ax, finals, metric, log):
    allv = np.concatenate([[r[1] for r in rows] for rows in finals.values()])
    allv = allv[np.isfinite(allv)]
    if log:
        lo = max(allv[allv > 0].min(), 1e-12) if (allv > 0).any() else 1e-12
        bins = np.logspace(np.log10(lo), np.log10(allv.max() + 1e-12), 21)
        ax.set_xscale("log")
    else:
        bins = np.linspace(allv.min(), allv.max() + 1e-12, 21)
    for m, rows in finals.items():
        st = style_for(m)
        ax.hist([r[1] for r in rows], bins=bins, color=st["color"], alpha=0.45,
                label=st["label"], histtype="stepfilled", edgecolor=st["color"])
    ax.set_ylabel("count")
    ax.legend(loc="upper right")


def _plot_scatter(ax, finals, log):
    rng = np.random.default_rng(0)
    for i, (m, rows) in enumerate(finals.items()):
        st = style_for(m)
        y = np.array([r[1] for r in rows])
        x = i + (rng.random(len(y)) - 0.5) * 0.3
        ax.scatter(x, y, color=st["color"], s=10, alpha=0.7, label=st["label"])
    ax.set_xticks(range(len(finals)))
    ax.set_xticklabels([style_for(m)["label"] for m in finals], rotation=30, ha="right")
    if log:
        ax.set_yscale("log")


def _plot_conv(ax, finals, log):
    for m, rows in finals.items():
        st = style_for(m)
        best = min(rows, key=lambda r: r[1])
        for cfg, _final, traj, epochs in rows:
            ax.plot(epochs, traj, color=st["color"], alpha=0.12, lw=0.6)
        # ax.plot(best[3], best[2], color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
    ax.set_xlabel("epoch")
    if log:
        ax.set_yscale("log")
    ax.legend(loc="upper right")


def plot_kkt(spec, methods=None, mode="cdf", metric="residual", tail=5,
             log=True, out="plots/kkt.pdf"):
    set_neurips_style()
    methods = methods or [m for m in METHODS if list_configs(spec, m)]
    finals = _finals_by_method(spec, methods, metric, tail)
    if not finals:
        print(f"no '{SPLIT}' metrics found under {spec.agg_root} "
              f"(run aggregate.py --approach opt first)")
        return None

    print(f"\n--- final {metric} (mean of last {tail} epochs), per method ---")
    for m, rows in finals.items():
        v = np.array([r[1] for r in rows])
        print(f"  {m:10s}: {len(v):3d} configs | min {v.min():.4g} | "
              f"median {np.median(v):.4g} | max {v.max():.4g}")

    rlabel = _AXLABEL.get(metric, metric)
    if mode == "all":
        fig, axes = plt.subplots(2, 2, figsize=(COL_WIDTH * 2, COL_WIDTH * 1.7))
        _plot_cdf(axes[0, 0], finals, log);      axes[0, 0].set_xlabel(rlabel)
        _plot_pdf(axes[0, 1], finals, metric, log); axes[0, 1].set_xlabel(rlabel)
        _plot_scatter(axes[1, 0], finals, log);  axes[1, 0].set_ylabel(rlabel)
        _plot_conv(axes[1, 1], finals, log);     axes[1, 1].set_ylabel(rlabel)
    else:
        fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.8))
        if mode == "cdf":
            _plot_cdf(ax, finals, log); ax.set_xlabel(rlabel)
        elif mode == "pdf":
            _plot_pdf(ax, finals, metric, log); ax.set_xlabel(rlabel)
        elif mode == "scatter":
            _plot_scatter(ax, finals, log); ax.set_ylabel(rlabel)
        elif mode == "conv":
            _plot_conv(ax, finals, log); ax.set_ylabel(rlabel)
        else:
            raise ValueError(f"unknown mode '{mode}'")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="../selection/opt/aggregated",
                    help="dir of aggregate.py --approach opt per-cell aggregates")
    ap.add_argument("--task", default="folktables_positive_rate_pair")
    ap.add_argument("--data", default="income")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--mode", default="cdf", choices=["cdf", "pdf", "scatter", "conv", "all"])
    ap.add_argument("--metric", default="residual",
                    choices=["residual", "grad_norm", "max_viol", "compl"])
    # ap.add_argument("--tail", type=int, default=1,
    #                 help="window (last K epochs) collapsed to each config's final value")
    ap.add_argument("--linear", action="store_true", help="linear metric axis (default: log)")
    ap.add_argument("--out", default="plots/kkt_cdf.pdf")
    args = ap.parse_args()
    spec = ExperimentSpec(name=args.task, task=args.task, data=args.data,
                          bound=args.bound, agg_root=args.agg)
    plot_kkt(spec, mode=args.mode, metric=args.metric, tail=1,
             log=not args.linear, out=args.out)
