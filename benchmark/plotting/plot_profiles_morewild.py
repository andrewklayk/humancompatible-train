"""plot_profiles_fair.py — Moré–Wild accuracy profiles over the fairness tasks.

Per (task, config, metric): s = (f_final - f_L) / (f_0 - f_L), where f_0 is the
config's epoch-0 value, f_final its tail-mean, and f_L the best final over ALL
methods/configs on that task. Panel = per-method CDF of s pooled over tasks.
Convention: f_0 <= f_L  =>  s = 0. Objective panel: feasible configs only.

Requires select_best.py's best_*.json when configs='best'.
"""
import json
from logging import config
import os
from re import split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prepare_results_pinn import ExperimentSpec, list_configs, metric_trajectory, _seed_frames
from plot_kkt import _metric_traj, _final_max_viol            # reuse, incl. bound handling
from plot_style import set_neurips_style, style_for, COL_WIDTH, top_legend
from plot_style import TEXT_WIDTH
from plot_PINNS import select_best_configs


_PANELS = [("loss", "Train loss"),
           ("kkt_viol", r"Feasibility $\max_j(c_j-b)_+$"),
           ("kkt_grad", r"Stationarity $\|\nabla_x L\|$"),
           ("kkt_compl", r"Complementarity $\sum_j|\lambda_j g_j|$")]   

def _pairs_by_method(spec, m, metric, tail, feas_tol=None):
    col = metric
    dfs = _seed_frames(spec, m)

    rows = []
    df = pd.concat(dfs).groupby(["config", "epoch"]).mean().reset_index()  # seed-mean
    if col not in df.columns:
        return None
    if metric == "kkt_viol":
        df[col] = df[col].clip(lower=0)

    if metric == "kkt_viol":
        df_feasible = df # for max constraint violation, include all
    else:
        # check if is feasible
        tail_mv = (df.sort_values("epoch").groupby("config")["kkt_viol"]
                        .apply(lambda s: s.tail(tail).mean()))
        df_feasible = df[df["config"].isin(tail_mv[tail_mv < 0.00011].index)]

        print(f"  [{spec.name}] {m}: {len(df_feasible['config'].unique())} feasible configs "
                f"out of {len(df['config'].unique())} total")
        
    # sort by epochs + mean the tail
    g = df_feasible.sort_values("epoch").groupby("config")
    finals = g[col].apply(lambda s: s.tail(tail).mean())                    # tail-mean

    rows = [(cfg, 
             float((df[(df['config'] == cfg) & (df['epoch'] == 0)][col]).iloc[0]) ,  # first value
             float(finals[cfg]) )
                for cfg, sub in g if cfg in finals.index]
    
    # add f0 and f0 to the infeasible rows
    infeasible_configs = set(df["config"].unique()) - set(df_feasible['config'].unique())
    for cfg in infeasible_configs:
        f0 = float((df[(df['config'] == cfg) & (df['epoch'] == 0)][col]).iloc[0])
        rows.append((cfg, f0, np.nan))

    return rows

def plot_profiles_pinns(specs, methods, configs="all", tail=5, tol_mult=1.0,
                       taus=None, out="plots/profiles_fair.pdf"):
    
    assert configs in ("all", "best")
    set_neurips_style()
    taus = np.logspace(-5, 0, 60) if taus is None else np.asarray(taus)
    
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH * 0.7, TEXT_WIDTH * 0.6),
                         sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (metric, title) in zip(axes, _PANELS):
        scores = {m: [] for m in methods}

        for spec in specs:
            print('', spec.name, metric)            
            
            per_method = {m: _pairs_by_method(spec, m, metric, tail, feas_tol=None) for m in methods}
            finals = [f for rows in per_method.values() for (_, _, f) in rows]

            if not finals:
                continue
            f_L = min(finals)                       # reference: full pool, always

            # get the best config 
            if configs == "best":
                best = select_best_configs(spec, methods, split="", best_validation_lastK=tail)

            for m in methods:
                keep = per_method[m]
                if configs == "best":
                    b = best[m]
                    keep = [r for r in keep if r[0] == b] if b is not None else []

                for cfg, f0, f in keep:

                    if f is np.nan:  # infeasible
                        continue

                    # compute relative constraint violation error 
                    if metric == "kkt_viol":
                        # _metric_traj already returns clip(c - b, 0); f is its tail-mean
                        scores[m].append(f / spec.bound)  
                    # compute Moré–Wild accuracy score
                    else:
                        gap = f0 - f_L
                        scores[m].append(0.0 if gap <= 0 else max(f - f_L, 0.0) / gap)
        
        # TODO: objective is wrong; it gives wierd results for PBM; do the same for fairness metrics

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

    # True is a running window mean; False is a tail
    running_average = False
    best_validation_window = 50

    names = ["E7", "E8", "E9"]
    specs = [ExperimentSpec(name="E7", data="helmholtz", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results"),
        ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results"),
        ExperimentSpec(name="E9", data="klein_gordon", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results")]

    methods = ["adam", "alm_proj", "pbm", "ssg"]

    plot_profiles_pinns(specs, methods, configs="all",
                       out="./results/plots/profiles_pinns_all.pdf")
    # plot_profiles_pinns(specs, methods, configs="best",
    #                    out="./results/plots/profiles_pinns_best.pdf")