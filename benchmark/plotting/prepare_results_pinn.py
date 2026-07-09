"""prepare_results_pinn.py — PINN-CSV backend for plot_kkt.py.
Same interface as prepare_results_plotting; reads runs_{method}.csv per seed,
averages across seeds, maps kkt_* column names."""
import os, re
import numpy as np
from aggregate_results import ExperimentSpec, _read_runs_csv   # reuse script 2's
import pandas as pd

_COLMAP = {"grad_norm": "kkt_grad", "max_viol": "kkt_viol",
           "compl": "kkt_compl", "loss": "loss"}                # opt-name -> pinn-name

def _seed_frames(spec, method):
    out = []
    for seed in spec.seeds:
        path = os.path.join(spec.seed_dir(seed), f"runs_{method}.csv")
        if os.path.exists(path):
            out.append(pd.read_csv(path))
    return out

def list_configs(spec, method):
    cfgs = set()
    for df in _seed_frames(spec, method):
        cfgs |= set(df["config"].unique())
    return sorted(cfgs)

def metric_trajectory(spec, method, cfg, split, metric):
    """(vals[L], None, None, epochs[L]) — seed-mean of one metric; None if absent.
    `split` ignored: PINN csv has no splits ('test' is a column, not a file)."""
    col = _COLMAP.get(metric, metric)
    per_seed = []
    epochs = None
    for df in _seed_frames(spec, method):
        sub = df[df["config"] == cfg].sort_values("epoch")
        if sub.empty or col not in sub.columns:
            continue
        per_seed.append(sub[col].to_numpy(dtype=float))
        epochs = sub["epoch"].to_numpy() if epochs is None else epochs
    if not per_seed:
        return None
    L = min(len(v) for v in per_seed)
    vals = np.stack([v[:L] for v in per_seed]).mean(axis=0)
    return vals, None, None, epochs[:L]

def dual_trajectory(spec, method, cfg, split):
    """(lam_mean[m,L], lam_std[m,L], epochs[L]) from lambda_j columns; None if absent."""
    per_seed, epochs = [], None
    for df in _seed_frames(spec, method):
        sub = df[df["config"] == cfg].sort_values("epoch")
        lcols = sorted([c for c in sub.columns if re.fullmatch(r"lambda_\d+", c)],
                       key=lambda s: int(s.split("_")[1]))
        if sub.empty or not lcols:
            continue
        per_seed.append(sub[lcols].to_numpy(dtype=float).T)     # (m, L)
        epochs = sub["epoch"].to_numpy() if epochs is None else epochs
    if not per_seed:
        return None
    L = min(c.shape[1] for c in per_seed)
    stacked = np.stack([c[:, :L] for c in per_seed])            # (seeds, m, L)
    return stacked.mean(0), stacked.std(0), epochs[:L]