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


_PANELS = [("loss", "Loss"),
           ("kkt_viol", r"Feasibility $\max_j(c_j-b)_+$"),
           ("kkt_grad", r"Stationarity $\|\nabla_x L\|$"),
           ("kkt_compl", r"Complementarity $\sum_j|\lambda_j g_j|$")]   


def _base(method):
    """('pbm__2' -> 'pbm', 2);  ('pbm' -> 'pbm', 1)."""
    if "__" in method:
        b, k = method.split("__"); return b, int(k)
    return method, 1

def _tol(spec, cutoff):
    """Relative feasibility slack off the bound (cutoff=1 loose, 2 tight)."""

    return 0.00011 if cutoff == 1 else 0.0001001 # 1 decimal vs 3 decimals

def _pairs_by_method(spec, m, metric, tail, feas_tol=None):
    col = metric
    base, cutoff = _base(m)
    dfs = _seed_frames(spec, base)

    rows = []
    df = pd.concat(dfs).groupby(["config", "epoch"]).mean().reset_index()  # seed-mean
    if col not in df.columns:
        return None
    if metric == "kkt_viol":
        df[col] = df[col].clip(lower=0)

    if metric == "kkt_viol" or metric == "kkt_compl" or metric == "kkt_grad":
        df_feasible = df # for max constraint violation, include all
    else:
        # check if is feasible
        tail_mv = (df.sort_values("epoch").groupby("config")["kkt_viol"]
                        .apply(lambda s: s.tail(tail).mean()))
        df_feasible = df[df["config"].isin(tail_mv[tail_mv < _tol(spec, cutoff)].index)]

        print(f"  [{spec.name}] {base}: {len(df_feasible['config'].unique())} feasible configs "
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
        rows.append((cfg, f0, np.inf))

    return rows


def _pairs_obj_feasibility(spec, m, tail):
    dfs = _seed_frames(spec, m)
    if not dfs:
        return []
    df = pd.concat(dfs).groupby(["config", "epoch"]).mean().reset_index()
    g = df.sort_values("epoch").groupby("config")[["loss", "kkt_viol"]]
    finals = g.apply(lambda s: s.tail(tail).mean())          # DataFrame: index=config
    return [(float(r["kkt_viol"]), float(r["loss"])) for _, r in finals.iterrows()]

def plot_tradeoff_scatter(specs, methods, tail=5, out="plots/pinn_tradeoff.pdf"):
    set_neurips_style()
    fig, axes = plt.subplots(1, len(specs), figsize=(COL_WIDTH*len(specs), COL_WIDTH*0.9))
    for ax, spec in zip(np.atleast_1d(axes), specs):
        for m in methods:
            m = _base(m)[0]  # only plot base methods once
            pts = _pairs_obj_feasibility(spec, m, tail)
            if not pts: continue
            st = style_for(m)
            v, f = zip(*pts)
            ax.scatter(v, f, s=8, alpha=0.6, color=st["color"], label=st["label"])
        ax.axvline(spec.bound, color="red", ls=":", lw=0.8)   # feasibility threshold
        ax.set_xscale("symlog", linthresh=spec.bound); ax.set_yscale("log")
        ax.set_title(spec.name); ax.set_xlabel(r"final $\max_j(c_j-b)$")
    np.atleast_1d(axes)[0].set_ylabel("final loss")
    top_legend(fig, np.atleast_1d(axes)[0])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    
    print(f"wrote {out}")

def plot_profiles_pinns(specs, methods, configs="all", tail=5, tol_mult=1.0,
                       taus=None, out="plots/profiles_fair.pdf"):
    
    assert configs in ("all", "best")
    set_neurips_style()
    taus = np.logspace(-5, 0, 60) if taus is None else np.asarray(taus)
    
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH * 0.7, TEXT_WIDTH * 0.6),
                        sharex=True, sharey=True)
        
    axes = axes.ravel()

    for i, (ax, (metric, title)) in enumerate(zip(axes, _PANELS)):
        ax.set_title(f"({chr(97 + i)}) {title}")
        feas_only = metric == "loss"
        panel_methods = methods if feas_only else list(dict.fromkeys(_base(m)[0] for m in methods))
        scores = {m: [] for m in panel_methods}

        for spec in specs:
            print('', spec.name, metric)            
            
            per_method = {m: _pairs_by_method(spec, m, metric, tail, feas_tol=None) for m in panel_methods}
            base_finals = {}
            for m, rows in per_method.items():
                base_finals.setdefault(_base(m)[0], []).extend(f for _, _, f in rows)
            
            finals = [f for v in base_finals.values() for f in v]

            if not finals:
                continue
            f_L = np.min(np.array(finals))                       # reference: full pool, always

            # get the best config 
            if configs == "best":
                best = select_best_configs(spec, panel_methods, split="", best_validation_lastK=tail)

            for m in panel_methods:
                keep = per_method[m]
                if configs == "best":
                    b = best[m]
                    keep = [r for r in keep if r[0] == b] if b is not None else []

                for cfg, f0, f in keep:

                    if not np.isfinite(f):  # infeasible marker
                        continue

                    # compute relative constraint violation error 
                    if metric == "kkt_viol":
                        # _metric_traj already returns clip(c - b, 0); f is its tail-mean
                        scores[m].append(f / spec.bound)  
                    # compute Moré–Wild accuracy score
                    else:
                        gap = f0 - f_L
                        scores[m].append(0.0 if gap <= 0 else max(f - f_L, 0.0) / gap)
        
        for m in panel_methods:
            s = np.sort(scores[m])
            if not len(s): continue
            base, cutoff = _base(m); st = style_for(base)
            ls = "-" if cutoff == 1 else "--"
            y = (s[None, :] <= taus[:, None]).mean(axis=1)
            ax.plot(taus, y, color=st["color"], ls=ls,
                    label=st["label"] if (cutoff == 1 or not feas_only) else None)
            print(f"  [{title}] {m}: {len(s)} pairs, "
                  f"frac(tau=1e-1)={np.mean(s <= 1e-1):.2f}")

        ax.set_xscale("log")
        ax.set_xlabel(r"accuracy $\tau$")
        # ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction of configs")
    axes[1].set_ylabel("fraction of configs")
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
                              results_root="results2"),
        ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results2"),
        ExperimentSpec(name="E9", data="klein_gordon", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results2")]

    methods = ["adam", "alm_proj__1", "alm_proj__2", "pbm__1", "pbm__2", "ssg__1", "ssg__2"]

    plot_profiles_pinns(specs, methods, configs="all",
                       out="./results/plots/profiles_pinns_all.pdf")
    # plot_profiles_pinns(specs, methods, configs="best",
    #                    out="./results/plots/profiles_pinns_best.pdf")

    plot_tradeoff_scatter(specs, methods, tail=best_validation_window, out="./results/plots/pinn_tradeoff.pdf")