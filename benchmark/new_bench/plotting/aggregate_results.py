"""aggregate_results.py (new_bench layout) — reads selection/aggregated/*.json.

Single source of truth: the seed-averaged, per-split, per-constraint curves that
`select_best.py` persists under `selection/aggregated/`. No re-scanning of the raw
multirun tree -- run `select_best.py` first.

Each `<cell>__cfgNNN.json` holds:
    task, data, algorithm, config_index, m, bound, select_filter, hyperparameters,
    splits: { "train"|"val"|"test": [ {epoch, loss_mean, loss_std, maxc_mean,
              maxc_std, c_0_mean, c_0_std, ..., n_seeds}, ... ] }

Public API (shape preserved so the plot scripts barely move):
    aggregate_experiment(spec, methods, split, tail, last_epoch) -> {method: DataFrame}
        per-config scalar table: config, loss_mean, loss_std,
        violation_constr_mean, violation_constr_std, feasible_mean_viol, n_seeds_found
    config_trajectory(spec, method, config_index, split)
        -> (loss_mean[L], loss_std[L], cons_mean[m,L]|None, cons_std[m,L]|None) | None

Note on statistics (vs the old raw-scanning backend): the central value of a
collapsed scalar is unchanged (means commute), but its error bar is now the
seed-std taken from the stored per-epoch curve (collapsed the same way as the
mean) rather than the std of a per-seed collapse. Feasibility is reported on the
seed-mean violation (`feasible_mean_viol`), matching select_best.py's rule.
"""
import glob
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# Single source of truth for the curve collapse lives in select_best.py (the producer
# of these aggregates); import it so selection and plots share identical semantics.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from select_best import collapse as _collapse  # noqa: E402

DEFAULT_METHODS = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]


@dataclass
class ExperimentSpec:
    name: str                 # label for plots
    task: str                 # matches the aggregated JSON 'task', e.g. "folktables_positive_rate_pair"
    data: str                 # matches 'data', e.g. "income"
    bound: float              # feasibility threshold (for the plotted line / feasibility)
    agg_root: str = "selection/aggregated"   # dir holding <cell>__cfgNNN.json
    seeds: Optional[tuple] = None   # kept for API parity; aggregation is already seed-averaged
    pinns: bool = False             # PINNs use the old pipeline; kept for API parity


@lru_cache(maxsize=None)
def _discover(agg_root, task, data):
    """method -> {config_index: payload dict}. Cached on the (hashable) args."""
    out = {}
    for p in sorted(glob.glob(os.path.join(agg_root, "*.json"))):
        with open(p) as f:
            payload = json.load(f)
        if payload.get("task") != task or payload.get("data") != data:
            continue
        out.setdefault(payload["algorithm"], {})[int(payload["config_index"])] = payload
    return out


def _split_frame(payload, split):
    recs = payload.get("splits", {}).get(split)
    if not recs:
        return None
    return pd.DataFrame(recs).sort_values("epoch").reset_index(drop=True)


def aggregate_method(spec: ExperimentSpec, method: str, split: str = "val",
                     tail: int = 10, last_epoch: bool = True) -> Optional[pd.DataFrame]:
    """Per-config scalar table for one method, collapsed from its stored curves."""
    cell = _discover(spec.agg_root, spec.task, spec.data)
    if method not in cell:
        return None
    rows = []
    for idx, payload in sorted(cell[method].items()):
        df = _split_frame(payload, split)
        if df is None or "loss_mean" not in df.columns:
            continue
        c = _collapse(df, tail, last_epoch)
        rows.append({"config": idx, "loss_mean": c["loss_mean"], "loss_std": c["loss_std"],
                     "violation_constr_mean": c["viol_mean"],
                     "violation_constr_std": c["viol_std"],
                     "n_seeds_found": int(df["n_seeds"].max())})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["feasible_mean_viol"] = out["violation_constr_mean"] <= spec.bound
    return out


def aggregate_experiment(spec: ExperimentSpec, methods=DEFAULT_METHODS,
                         split: str = "val", tail: int = 10, last_epoch: bool = True) -> dict:
    """{method: per-config DataFrame}. Methods with no aggregated configs are skipped."""
    result = {}
    for m in methods:
        agg = aggregate_method(spec, m, split=split, tail=tail, last_epoch=last_epoch)
        if agg is None:
            print(f"  [{spec.name}] {m}: no aggregated configs found, skipping")
            continue
        n_feas = int(agg["feasible_mean_viol"].sum())
        print(f"  [{spec.name}] {m}: {len(agg)} configs, "
              f"{n_feas} feasible (mean viol <= {spec.bound}), "
              f"seeds/config ~{int(agg['n_seeds_found'].median())}")
        result[m] = agg
    return result


def config_trajectory(spec: ExperimentSpec, method: str, config_index: int, split: str):
    """Seed-averaged per-epoch curves for one config/split.

    Returns (loss_mean[L], loss_std[L], cons_mean[m,L]|None, cons_std[m,L]|None) or
    None if the config/split is absent. cons_* are None when the split has no
    constraint columns (e.g. an unconstrained image task).
    """
    cell = _discover(spec.agg_root, spec.task, spec.data)
    payload = cell.get(method, {}).get(int(config_index))
    if payload is None:
        return None
    df = _split_frame(payload, split)
    if df is None or "loss_mean" not in df.columns:
        return None
    loss_mean = df["loss_mean"].to_numpy()
    loss_std = df["loss_std"].to_numpy()
    cmean, cstd = [], []
    for i in range(int(payload.get("m", 0))):
        if f"c_{i}_mean" in df.columns:
            cmean.append(df[f"c_{i}_mean"].to_numpy())
            cstd.append(df[f"c_{i}_std"].to_numpy())
    cons_mean = np.vstack(cmean) if cmean else None
    cons_std = np.vstack(cstd) if cstd else None
    return loss_mean, loss_std, cons_mean, cons_std


def list_configs(spec: ExperimentSpec, method: str):
    """Sorted config indices available for one method in this cell (empty if none)."""
    cell = _discover(spec.agg_root, spec.task, spec.data)
    return sorted(cell.get(method, {}).keys())


def metric_trajectory(spec: ExperimentSpec, method: str, config_index: int,
                      split: str, metric: str):
    """Seed-averaged per-epoch curve of an arbitrary metric column.

    Returns (mean[L], std[L]|None, std_init[L]|None, epochs[L]) or None if the
    config / split / metric is absent. ``metric`` is the bare name (e.g. 'grad_norm',
    'max_viol', 'compl', 'L'); the stored columns are '<metric>_mean' etc. Used by the
    KKT plots over the 'opt' split.
    """
    cell = _discover(spec.agg_root, spec.task, spec.data)
    payload = cell.get(method, {}).get(int(config_index))
    if payload is None:
        return None
    df = _split_frame(payload, split)
    if df is None or f"{metric}_mean" not in df.columns:
        return None
    mean = df[f"{metric}_mean"].to_numpy()
    std = df[f"{metric}_std"].to_numpy() if f"{metric}_std" in df.columns else None
    std_init = df[f"{metric}_std_init"].to_numpy() if f"{metric}_std_init" in df.columns else None
    return mean, std, std_init, df["epoch"].to_numpy()


if __name__ == "__main__":
    import sys
    spec = ExperimentSpec(
        name="demo",
        task="folktables_positive_rate_pair",
        data="income",
        bound=0.1,
        agg_root=sys.argv[1] if len(sys.argv) > 1 else "selection/aggregated",
    )
    agg = aggregate_experiment(spec, split="val", tail=5)
    for method, df in agg.items():
        print(f"\n=== {method} ===")
        print(df.head().to_string(index=False))
