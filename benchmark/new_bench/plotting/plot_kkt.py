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
  duals    dual variables lambda_j vs epoch, one panel per constrained method, averaged
           over all configs of the method; unconstrained methods (adam/ssg) skipped

``--metric {residual,grad_norm,max_viol,compl,objective}`` plots a single quantity
instead (ignored by ``duals``, which always plots the lambda_j curves). ``objective``
is the optimization objective f (the loss), read from the TRAIN split -- opt.csv holds
only the KKT metrics, not the loss.

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

from prepare_results_plotting import (ExperimentSpec, list_configs, metric_trajectory,
                                      dual_trajectory)
from plot_style import set_neurips_style, style_for, COL_WIDTH

SPLIT = "opt"
METHODS = ["adam", "alm_proj", "pbm", "ssg"]
# METHODS = ["alm_proj", "pbm"]
_AXLABEL = {"residual": "KKT residual $r$", "grad_norm": r"$\|\nabla_x L\|$",
            "max_viol": "feasibility $\\max_j(c_j-b)_+$", "compl": "complementarity $\\sum_j|\\lambda_j g_j|$",
            "objective": "objective $f$ (loss)"}


_PANELS = [("grad_norm", "Stationarity Error, " + r"$\|\nabla_x L\|$"),
           ("max_viol", "Feasibility Error $\\max_j(c_j-b)_+$"),
           ("compl", "complementarity $\\sum_j|\\lambda_j g_j|$"),
           ("objective", "Train Loss")]

def _residual_traj(spec, method, cfg):
    """(r[L], epochs[L]) with r = ‖∇L‖ + relu(max_viol) + |compl|, or None."""
    gn = metric_trajectory(spec, method, cfg, SPLIT, "grad_norm")
    if gn is None:
        return None
    r = np.asarray(gn[0], dtype=float)
    mv = metric_trajectory(spec, method, cfg, SPLIT, "max_viol")
    if mv is not None:
        r = r + np.clip(np.asarray(mv[0], dtype=float) - spec.bound, 0, None)
    cp = metric_trajectory(spec, method, cfg, SPLIT, "compl")
    if cp is not None:
        r = r + np.abs(np.asarray(cp[0], dtype=float))
    return r, gn[3]


def _metric_traj(spec, method, cfg, metric):
    if metric == "residual":
        return _residual_traj(spec, method, cfg)
    if metric == "objective":
        # the optimization objective f is the loss, recorded on the TRAIN split
        # (opt.csv from evaluate_optimality holds only the KKT metrics, not the loss).
        t = metric_trajectory(spec, method, cfg, "train", "loss")
        return (np.asarray(t[0], dtype=float), t[3]) if t is not None else None
    t = metric_trajectory(spec, method, cfg, SPLIT, metric)
    if t is None:
        return None
    vals = np.asarray(t[0], dtype=float)
    if metric == "max_viol":
        vals = np.clip(vals - spec.bound, 0, None)
    return vals, t[3]


def _final_max_viol(spec, method, cfg, tail):
    """Config's final (last-``tail``-epoch mean) signed max violation maxⱼ(cⱼ-b), or
    None if the config has no max_viol column. <= 0 means feasible."""
    t = metric_trajectory(spec, method, cfg, SPLIT, "max_viol")
    if t is None or len(t[0]) == 0:
        return None
    return float(np.mean(np.asarray(t[0], dtype=float)[-tail:])) - spec.bound


def _collect_final(spec, method, metric, tail, feas_tol=None):
    """[(cfg, final_scalar, traj, epochs)] over all configs of a method (skips missing).
    When ``feas_tol`` is set (objective plots), drops configs whose final max violation
    exceeds it -- an infeasible config can reach a low objective by ignoring the
    constraints, so it isn't comparable."""
    out = []
    for cfg in list_configs(spec, method):
        t = _metric_traj(spec, method, cfg, metric)
        if t is None or len(t[0]) == 0:
            continue
        if feas_tol is not None:
            mv = _final_max_viol(spec, method, cfg, tail)
            if mv is None or mv > feas_tol:
                continue
        out.append((cfg, float(np.mean(t[0][-tail:])), t[0], t[1]))
    return out


def _finals_by_method(spec, methods, metric, tail, feas_tol=None):
    d = {}
    for m in methods:
        rows = _collect_final(spec, m, metric, tail, feas_tol=feas_tol)
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


def _mean_dual_trajectory(spec, method):
    """Config-averaged dual trajectory for a method: (lambda_mean[m,L], lambda_std[m,L],
    epochs[L]), where each lambda_j is averaged over all configs of the method (each
    config already seed-averaged) and lambda_std is the spread ACROSS configs. None if
    the method stores no duals. Configs are truncated to their common epoch length."""
    curves, epochs = [], None
    for cfg in list_configs(spec, method):
        dt = dual_trajectory(spec, method, cfg, SPLIT)
        if dt is None:
            continue
        curves.append(dt[0])                # lambda_mean[m, L] for this config
        epochs = dt[2] if epochs is None else epochs
    if not curves:
        return None
    L = min(c.shape[1] for c in curves)
    stacked = np.stack([c[:, :L] for c in curves], axis=0)  # (n_config, m, L)
    return stacked.mean(axis=0), stacked.std(axis=0), epochs[:L]


def _plot_duals(spec, methods, out="plots/kkt_duals.pdf"):
    """One panel per constrained method: each dual variable lambda_j vs epoch, averaged
    over all configs of the method. Methods with no lambda_j (adam/ssg) are skipped.
    With few constraints (<=10) each lambda_j gets a distinct color + a legend and a
    +-across-configs band; with many, lambda_j is shaded along a sequential colormap
    keyed by index j (shared colorbar, no bands) so 30+ curves stay legible."""
    panels = []
    for m in methods:
        mt = _mean_dual_trajectory(spec, m)
        if mt is not None:
            panels.append((m, mt))
    if not panels:
        print(f"no dual variables found under {spec.agg_root} "
              f"(only constrained methods store lambda_j)")
        return None

    m_max = max(lam_mean.shape[0] for _, (lam_mean, _, _) in panels)
    many = m_max > 10                       # too many for a per-line legend
    norm = plt.Normalize(vmin=0, vmax=max(m_max - 1, 1))
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(COL_WIDTH * len(panels), COL_WIDTH * 0.85))
    for ax, (m, (lam_mean, lam_std, epochs)) in zip(axes[0], panels):
        for j in range(lam_mean.shape[0]):
            color = plt.cm.viridis(norm(j)) if many else plt.cm.tab10(j % 10)
            ax.plot(epochs, lam_mean[j], color=color, lw=1.0,
                    label=None if many else rf"$\lambda_{{{j}}}$")
            if not many:                    # bands only readable for a handful of duals
                ax.fill_between(epochs, lam_mean[j] - lam_std[j], lam_mean[j] + lam_std[j],
                                color=color, alpha=0.15, linewidth=0)
        ax.set_title(style_for(m)["label"])
        ax.set_xlabel("epoch")
        if not many:
            ax.legend(loc="best", ncol=2, fontsize="x-small")
    axes[0, 0].set_ylabel(r"dual variable $\lambda_j$ (mean over configs)")

    if many:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        cbar = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.02, pad=0.01)
        cbar.set_label(r"constraint index $j$")
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


def plot_kkt(spec, methods=None, mode="cdf", metric="residual", tail=5,
             log=True, out="plots/kkt.pdf", feas_tol=0.0):
    set_neurips_style()
    methods = [m for m in METHODS if list_configs(spec, m)] if methods is None else methods
    if mode == "duals":
        return _plot_duals(spec, methods, out=out)

    # Only the objective (loss) is filtered for feasibility: a low loss is meaningless
    # if the config is infeasible. The KKT metrics already encode feasibility themselves.
    feas = feas_tol if metric == "objective" else None
    finals = _finals_by_method(spec, methods, metric, tail, feas_tol=feas)

    if not finals:
        extra = f" feasible at max_viol<={feas_tol}" if feas is not None else ""
        print(f"no '{SPLIT}' metrics found under {spec.agg_root}{extra} "
              f"(run aggregate.py --approach opt first)")
        return None

    if feas is not None:
        print(f"[objective] keeping only configs with final max_viol <= {feas_tol}")
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



def plot_kkt_boxes(spec, methods=None, tail=5, out="plots/kkt_boxes.pdf", 
                log_scales = [True, False, False, False], feas_tol=0.0, metric="objective"):
    """3 panels (stationarity | feasibility | KKT residual), one box per method;
    each box = distribution of final (tail-mean) values over configs."""
    set_neurips_style()
    methods = METHODS if methods is None else methods
    fig, axes = plt.subplots(1, 4, figsize=(COL_WIDTH * 3, COL_WIDTH * 0.9))
    feas = feas_tol if metric == "objective" else None
    
    idx = 0
    for ax, (metric, title) in zip(axes, _PANELS):
        finals = _finals_by_method(spec, methods, metric, tail)

        # return an error if no data found
        if not finals:
            extra = f" feasible at max_viol<={feas_tol}" if feas is not None else ""
            print(f"no '{SPLIT}' metrics found under {spec.agg_root}{extra} "
                f"(run aggregate.py --approach opt first)")
            return None

        # aggregate the data for this feature
        ms = [m for m in methods if m in finals]
        data = [[r[1] for r in finals[m]] for m in ms]
        if not data:
            continue
        ax.boxplot(data, tick_labels=[style_for(m)["label"] for m in ms],
                   flierprops=dict(markersize=2))

        if log_scales[idx]:
            ax.set_yscale("log"); 
            axes[idx].set_ylabel("Error (log scale)")
        else: 
            axes[idx].set_ylabel("Error")

        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        idx += 1

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_kkt_boxes_single(specs, methods=None, tail=5, out="plots/kkt_boxes.pdf", 
                log_scales = [True, False, False, False], feas_tol=0.0, metric="objective"):
    """3 panels (stationarity | feasibility | KKT residual), one box per method;
    each box = distribution of final (tail-mean) values over configs."""
 
    set_neurips_style()
    methods = METHODS if methods is None else methods
    fig, axes = plt.subplots(1, 4, figsize=(COL_WIDTH * 3, COL_WIDTH * 0.9))

    # for metric
    for ax, (metric, title) in zip(axes, _PANELS):
        pooled = {m: [] for m in methods}

        for spec in specs:
            finals = _finals_by_method(spec, methods, metric, tail)
            # finals = _finals_by_method(spec, methods, metric, tail, feas_tol=feas)
            ms = [m for m in methods if m in finals]

            # per-spec baseline: best (lowest) final across ALL methods' runs
            all_vals = [r[1] for m in ms for r in finals[m]]
            best = min(all_vals)

            eps = 1e-4
            for m in ms:
                pooled[m].extend((r[1] + eps) / (best + eps) for r in finals[m])

        ms_plot = [m for m in methods if pooled[m]] 
        data = [pooled[m] for m in ms_plot]
        ax.boxplot(data, tick_labels=[style_for(m)["label"] for m in ms_plot],
                   flierprops=dict(markersize=2))
        ax.tick_params(axis="x", rotation=30)
        ax.set_yscale("log")          # ratios spanning orders of magnitude read better on log
        ax.axhline(1.0, ls="--", lw=0.6, color="gray")   # the "best" reference line
        ax.set_title(title)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")

if __name__ == "__main__":

    # all possible experiments
    experiments = [ 'folktables_positive_rate_vec',
                    'folktables_positive_rate_pair', 
                    'dutch_positive_rate_pair']

    data_map = {    "folktables_positive_rate_vec": "income", 
                    "folktables_positive_rate_pair": "income",
                    "dutch_positive_rate_pair": "dutch"
    }

    bounds_map = {
                    "folktables_positive_rate_vec": 0.2, 
                    "folktables_positive_rate_pair": 0.1,
                    "dutch_positive_rate_pair": 0.1
    }

    # map to the E 
    mapping_name = {"folktables_positive_rate_vec": "E2", 
                    "folktables_positive_rate_pair": "E3",
                    "dutch_positive_rate_pair": "E4"}

    # define output folder
    out = "../../results/plots/"
    agg = "../selection/aggregated/"

    specs = []

    # loop over all experiments and create the experiments
    for experiment in experiments: 
        
        # load the details about the experiment
        task = experiment
        data = data_map[experiment]
        bound = bounds_map[experiment]

        spec = ExperimentSpec(name=task , task=task, data=data,
                        bound=bound, agg_root=agg)

        specs.append(spec)
                          

    # plot all
    for spec in specs: 
        plot_kkt_boxes(spec, out=out + f"kkt_{mapping_name[spec.task]}.pdf")

    # plot a single combined plot
    # plot_kkt_boxes_single(specs, methods=None, tail=5, out="../../results/plots/kkt_fair.pdf", 
    #             feas_tol=0.0, metric="objective")

