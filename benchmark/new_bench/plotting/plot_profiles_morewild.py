"""plot_profiles_fair.py — Moré–Wild accuracy profiles over the fairness tasks.

Per (task, config, metric): s = (f_final - f_L) / (f_0 - f_L), where f_0 is the
config's epoch-0 value, f_final its tail-mean, and f_L the best final over ALL
methods/configs on that task. Panel = per-method CDF of s pooled over tasks.
Convention: f_0 <= f_L  =>  s = 0. Objective panel: feasible configs only.

Requires select_best.py's best_*.json when configs='best'.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from prepare_results_plotting import ExperimentSpec, list_configs, metric_trajectory
from plot_kkt import _metric_traj, _final_max_viol            # reuse, incl. bound handling
from plot_style import set_neurips_style, style_for, COL_WIDTH, top_legend
from plot_style import TEXT_WIDTH


_PANELS = [("objective", "Train loss"),
           ("max_viol", r"Feasibility $\max_j(c_j-b)_+$"),
           ("grad_norm", r"Stationarity $\|\nabla_x L\|$"),
           ("compl", r"Complementarity $\sum_j|\lambda_j g_j|$")]


def _best_config(spec, method, tol_mult=1.0):
    """Winning config index from select_best.py's best_*.json, or None."""
    sel_dir = os.path.dirname(os.path.abspath(spec.agg_root).rstrip("/"))
    base = os.path.join(sel_dir, f"best_{spec.task}_{spec.data}_{method}")
    for p in (f"{base}__tol{tol_mult:g}.json", f"{base}__none.json", f"{base}.json"):
        if os.path.exists(p):
            return int(json.load(open(p))["config_index"])
    return None


def _pairs(spec, method, metric, tail, feas_only):
    """[(f0, f_final)] over configs of one (task, method, metric)."""
    out = []
    for cfg in list_configs(spec, method):
        if feas_only:
            mv = _final_max_viol(spec, method, cfg, tail)
            if 'weight' in spec.task:
                if mv is None or mv >= 0.1:
                    continue
            else: 
                if mv is None or mv >= 0.01:
                    continue
        t = _metric_traj(spec, method, cfg, metric)
        if t is None or len(t[0]) == 0:
            continue
        out.append((cfg, float(t[0][0]), float(np.mean(t[0][-tail:]))))
    return out


def plot_profiles_fair(specs, methods, configs="all", tail=5, tol_mult=1.0,
                       taus=None, out="plots/profiles_fair.pdf"):
    assert configs in ("all", "best")
    set_neurips_style()
    taus = np.logspace(-5, 0, 60) if taus is None else np.asarray(taus)
    
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH * 0.7, TEXT_WIDTH * 0.6),
                         sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, _PANELS):
        feas_only = metric == "objective" 
        scores = {m: [] for m in methods}

        for spec in specs:
            per_method = {m: _pairs(spec, m, metric, tail, feas_only) for m in methods}
            finals = [f for rows in per_method.values() for (_, _, f) in rows]

            if not finals:
                continue
            f_L = min(finals)                       # reference: full pool, always

            for m in methods:
                keep = per_method[m]
                if configs == "best":
                    b = _best_config(spec, m, tol_mult)
                    keep = [r for r in keep if r[0] == b] if b is not None else []

                for cfg, f0, f in keep:
                    # compute relative constraint violation error 
                    if metric == "max_viol":
                        # _metric_traj already returns clip(c - b, 0); f is its tail-mean
                        scores[m].append(f / spec.bound)  
                    # compute Moré–Wild accuracy score
                    else:
                        gap = f0 - f_L
                        scores[m].append(0.0 if gap <= 0 else max(f - f_L, 0.0) / gap)

        for m in methods:
            s = np.sort(scores[m])
            if not len(s):
                continue
            st = style_for(m)
            y = (s[None, :] <= taus[:, None]).mean(axis=1)
            ax.plot(taus, y, color=st["color"], ls=st["ls"], label=st["label"])
            print(f"  [{title}] {m}: {len(s)} pairs, "
                  f"frac(tau=1e-1)={np.mean(s <= 1e-1):.2f}")

        ax.set_xscale("log")
        ax.set_xlabel(r"accuracy $\tau$")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction of configs")
    top_legend(fig, axes[0])

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    
    experiments = [ 'weight_norm',
                    'folktables_positive_rate_vec',
                    'folktables_positive_rate_pair', 
                    'dutch_positive_rate_pair']

    data_map = {    "weight_norm": "income_norm",
                    "folktables_positive_rate_vec": "income", 
                    "folktables_positive_rate_pair": "income",
                    "dutch_positive_rate_pair": "dutch"
    }
    bounds_map = {  "weight_norm": 2.0,
                    "folktables_positive_rate_vec": 0.2, 
                    "folktables_positive_rate_pair": 0.1,
                    "dutch_positive_rate_pair": 0.1
    }

    agg = "../selection/aggregated/"
    specs = [ExperimentSpec(name=e, task=e, data=data_map[e],
                            bound=bounds_map[e], agg_root=agg)
             for e in experiments]
    methods = ["adam", "alm_proj", "pbm", "ssg"]

    plot_profiles_fair(specs, methods, configs="all",
                       out="../../results/plots/profiles_fair_all.pdf")
    plot_profiles_fair(specs, methods, configs="best",
                       out="../../results/plots/profiles_fair_best.pdf")