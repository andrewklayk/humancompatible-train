"""prepare_results_plotting.py — plotting data backend over aggregate.py's output.

Reads the per-cell aggregates `aggregate.py` writes under `selection/aggregated/`
(one `<cell>.json` metadata + `<cell>.csv` long curves per cell); never re-scans the
raw multirun tree, so run `aggregate.py` first. Best-config selection is a separate
step (`select_best.py`, used only by plot_fair).

Public API (consumed by the plot_* scripts):
    ExperimentSpec                                    -- name, task, data, bound, agg_root
    aggregate_experiment(spec, methods, split, tail)  -> {method: per-config DataFrame}
        columns: config, loss_mean, loss_std, violation_constr_mean,
                 violation_constr_std, n_seeds_found, feasible_mean_viol
    config_trajectory(spec, method, cfg, split)
        -> (loss_mean[L], loss_std[L], cons_mean[m,L]|None, cons_std[m,L]|None) | None
    metric_trajectory(spec, method, cfg, split, metric)
        -> (mean[L], std[L]|None, std_init[L]|None, epochs[L]) | None
    dual_trajectory(spec, method, cfg, split)
        -> (lambda_mean[m,L], lambda_std[m,L], epochs[L]) | None
    config_params(spec, method, cfg) -> {dotted_hparam: value}   (e.g. 'moreau.mu': 0.0)
    list_configs(spec, method, where=None) -> [config_index, ...]
        where: {dotted_hparam: value|predicate} keeps only matching configs
"""
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

# `collapse` (curve -> representative scalar) lives in select_best.py, the selector;
# import it so selection and plots use identical windowing semantics.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from select_best import collapse as _collapse  # noqa: E402

DEFAULT_METHODS = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]


@dataclass
class ExperimentSpec:
    name: str                                # label for plots / logs
    task: str                                # aggregated 'task', e.g. "folktables_positive_rate_pair"
    data: str                                # aggregated 'data', e.g. "income"
    bound: float                             # feasibility threshold
    agg_root: str = "selection/aggregated"   # dir of aggregate.py's <cell>.csv + <cell>.json


@lru_cache(maxsize=None)
def _load(agg_root, task, data):
    """(task, data) -> {method: {config_index: curves DataFrame}}. Reads every
    <cell>.json matching (task, data) and its sibling <cell>.csv, split by config."""
    out = {}
    for json_path in sorted(glob.glob(os.path.join(agg_root, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("task") != task or meta.get("data") != data:
            continue
        curves = pd.read_csv(json_path[:-5] + ".csv")
        out[meta["algorithm"]] = {int(i): g for i, g in curves.groupby("config")}
    return out


def _flatten(d, prefix=""):
    """Flatten a nested hyperparameters dict to {dotted_key: value}; non-dict leaves
    (incl. lists like penalty_range) are kept as-is. E.g. {'primal': {'lr': 0.01}} ->
    {'primal.lr': 0.01}."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix=key + "."))
        else:
            flat[key] = v
    return flat


@lru_cache(maxsize=None)
def _load_hparams(agg_root, task, data):
    """(task, data) -> {method: {config_index: flat_hparams}}. Reads each matching
    <cell>.json's `configs` list and flattens hyperparameters to dotted keys."""
    out = {}
    for json_path in sorted(glob.glob(os.path.join(agg_root, "*.json"))):
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("task") != task or meta.get("data") != data:
            continue
        out[meta["algorithm"]] = {int(c["config_index"]): _flatten(c.get("hyperparameters", {}))
                                  for c in meta.get("configs", [])}
    return out


def _curve(spec, method, config_index, split):
    """Per-epoch DataFrame for one (method, config, split), epoch-sorted, or None."""
    df = _load(spec.agg_root, spec.task, spec.data).get(method, {}).get(int(config_index))
    if df is None:
        return None
    df = df[df["split"] == split]
    if df.empty:
        return None
    return df.drop(columns=["config", "split"]).sort_values("epoch").reset_index(drop=True)


def config_params(spec, method, config_index):
    """Flattened dotted-key hyperparameters for one config (e.g. {'primal.lr': 0.01,
    'moreau.mu': 0.0, 'dual.penalty_mult': 0.5}), or {} if unavailable. Handy for
    labeling filtered curves."""
    return _load_hparams(spec.agg_root, spec.task, spec.data).get(method, {}).get(int(config_index), {})


def list_configs(spec, method, where=None):
    """Sorted config indices for one method (empty if none). If ``where`` is given, keep
    only configs whose (flattened, dotted) hyperparameters match EVERY entry: map a
    dotted key to a value for equality (e.g. ``{'moreau.mu': 0.0}``) or to a callable
    predicate for ranges (e.g. ``{'primal.lr': lambda v: v < 0.02}``). Keys may be given
    with or without an 'algorithm.' prefix; a key absent from a config excludes it."""
    idx = sorted(_load(spec.agg_root, spec.task, spec.data).get(method, {}))
    if not where:
        return idx
    hp = _load_hparams(spec.agg_root, spec.task, spec.data).get(method, {})

    def match(ci):
        params = hp.get(ci, {})
        for key, cond in where.items():
            k = key[len("algorithm."):] if key.startswith("algorithm.") else key
            if k not in params:
                return False
            val = params[k]
            if not (cond(val) if callable(cond) else val == cond):
                return False
        return True

    return [ci for ci in idx if match(ci)]


def aggregate_method(spec, method, split="val", tail=10, last_epoch=True):
    """Per-config scalar table for one method: each config's curve collapsed to a
    representative (loss, violation). None if the method has no such configs."""
    records = []
    for config_index in list_configs(spec, method):
        curve = _curve(spec, method, config_index, split)
        if curve is None or "loss_mean" not in curve.columns:
            continue
        c = _collapse(curve, tail, last_epoch)
        records.append({"config": config_index,
                        "loss_mean": c["loss_mean"], "loss_std": c["loss_std"],
                        "violation_constr_mean": c["viol_mean"],
                        "violation_constr_std": c["viol_std"],
                        "n_seeds_found": int(curve["n_seeds"].max())})
    if not records:
        return None
    table = pd.DataFrame(records)
    table["feasible_mean_viol"] = table["violation_constr_mean"] <= spec.bound
    return table


def aggregate_experiment(spec, methods=DEFAULT_METHODS, split="val", tail=10, last_epoch=True):
    """{method: per-config DataFrame}; methods with no aggregated configs are skipped."""
    per_method = {}
    for method in methods:
        table = aggregate_method(spec, method, split=split, tail=tail, last_epoch=last_epoch)
        if table is None:
            print(f"  [{spec.name}] {method}: no aggregated configs, skipping")
            continue
        n_feasible = int(table["feasible_mean_viol"].sum())
        print(f"  [{spec.name}] {method}: {len(table)} configs, {n_feasible} feasible "
              f"(mean viol <= {spec.bound}), seeds/config ~{int(table['n_seeds_found'].median())}")
        per_method[method] = table
    return per_method


def config_trajectory(spec, method, config_index, split):
    """Seed-averaged per-epoch (loss_mean, loss_std, cons_mean[m,L], cons_std[m,L]) for
    one config/split, or None. cons_* are None when the split has no constraint columns."""
    curve = _curve(spec, method, config_index, split)
    if curve is None or "loss_mean" not in curve.columns:
        return None
    c_cols = sorted((c[:-5] for c in curve.columns if re.fullmatch(r"c_\d+_mean", c)),
                    key=lambda s: int(s.split("_")[1]))

    def stack(suffix):
        rows = [curve[f"{c}_{suffix}"].to_numpy() for c in c_cols]
        return np.vstack(rows) if rows else None

    return curve["loss_mean"].to_numpy(), curve["loss_std"].to_numpy(), stack("mean"), stack("std")


def metric_trajectory(spec, method, config_index, split, metric):
    """Seed-averaged per-epoch curve of an arbitrary metric: (mean, std|None,
    std_init|None, epochs), or None if the config/split/metric is absent. Used by the
    KKT plots over the 'opt' split (metric e.g. 'grad_norm', 'max_viol', 'compl')."""
    curve = _curve(spec, method, config_index, split)
    if curve is None or f"{metric}_mean" not in curve.columns:
        return None

    def col(name):
        return curve[name].to_numpy() if name in curve.columns else None

    return col(f"{metric}_mean"), col(f"{metric}_std"), col(f"{metric}_std_init"), curve["epoch"].to_numpy()


def dual_trajectory(spec, method, config_index, split="opt"):
    """Seed-averaged per-epoch dual variables for one config/split:
    (lambda_mean[m,L], lambda_std[m,L], epochs[L]), or None if the split has no
    lambda_j columns (unconstrained methods / SSG store no duals). m = number of
    constraints, rows ordered lambda_0..lambda_{m-1}."""
    curve = _curve(spec, method, config_index, split)
    if curve is None:
        return None
    lam = sorted((c[:-5] for c in curve.columns if re.fullmatch(r"lambda_\d+_mean", c)),
                 key=lambda s: int(s.split("_")[1]))
    if not lam:
        return None

    def stack(suffix):
        return np.vstack([curve[f"{c}_{suffix}"].to_numpy() for c in lam])

    return stack("mean"), stack("std"), curve["epoch"].to_numpy()


if __name__ == "__main__":
    spec = ExperimentSpec("demo", "folktables_positive_rate_pair", "income", 0.1,
                          agg_root=sys.argv[1] if len(sys.argv) > 1 else "selection/aggregated")
    for method, table in aggregate_experiment(spec, split="val", tail=5).items():
        print(f"\n=== {method} ===\n{table.head().to_string(index=False)}")
