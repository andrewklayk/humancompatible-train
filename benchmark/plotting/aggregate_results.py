"""
aggregate_results.py

Loads gridsearch CSVs across seeds and produces, per (experiment, method), a tidy
per-config table whose loss/violation are MEANED across seeds. This is the shared
input for both the robustness CDF and the convergence plots.

Directory contract (from run_gridsearch.py + the batch script):
    results/{data}_{task}{seed}/runs_{method}_train.csv
    results/{data}_{task}{seed}/runs_{method}_val.csv
    results/{data}_{task}{seed}/best_{method}.json        (optional)

CSV contract (from runs_to_df): a leading UNNAMED column = config index, then
    epoch, time, loss, acc, c_0 ... c_{m-1}
The real row identity is (config_idx, epoch). 'acc' is a stringified array (ignored).

Feasibility is defined ONE-SIDED to match what the optimizer enforces (g(x) <= eps):
    max over signed c_i  <=  bound
(NOT max|c_i| — see the long discussion; the signed +/- pairs make one-sided correct.)
"""

import os
import re
import json
import glob
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── one experiment's spec: how to find its dirs + its feasibility bound ───────
@dataclass
class ExperimentSpec:
    name: str                 # label for plots, e.g. "income_pairwise"
    data: str                 # the {data} part of the dir name, e.g. "income"
    task: str                 # the {task} part, e.g. "folktables_positive_rate_pair"
    bound: float              # feasibility threshold (constraint_cfg['bound'])
    pinns: bool = False
    seeds: tuple = (0, 1, 2)  # which seeds to aggregate
    results_root: str = "results"

    def seed_dir(self, seed: int) -> str:
        return os.path.join(self.results_root, f"{self.data}_{self.task}{seed}")


# ── per-config, per-seed extraction ──────────────────────────────────────────
def _read_runs_csv(path: str) -> pd.DataFrame:
    """Read a runs_*.csv and return it with a proper 'config' column.

    The first column is unnamed (the config index from runs_to_df's keys=); pandas
    names it 'Unnamed: 0'. We rename it to 'config'.
    """
    df = pd.read_csv(path)
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.rename(columns={first: "config"})
    elif "config" not in df.columns:
        # fallback: assume the first column is the config index regardless of name
        df = df.rename(columns={first: "config"})
    return df

def _rolling_min_per_config(df, bound, pinns, tail=5):
    c_cols = [c for c in df.columns if re.fullmatch(r"c_\d+", c)]
    if not c_cols:
        raise ValueError("no constraint columns")
    rows = []
    for _, g in df.sort_values("epoch").groupby("config"):
        key = "val" if pinns else "loss"
        smooth = g[key].rolling(tail, min_periods=1).mean()
        rows.append(g.loc[smooth.idxmin()])
    last = pd.DataFrame(rows)
    last["max_viol"] = last[c_cols].max(axis=1)
    last["feasible"] = last["max_viol"] <= bound
    if pinns:
        return last[["config","loss","max_viol","test","val","feasible"]].reset_index(drop=True)
    return last[["config","loss","max_viol","feasible"]].reset_index(drop=True)

def _last_epoch_per_config(df: pd.DataFrame, bound: float,  pinns: bool, tail=10) -> pd.DataFrame:
    """Collapse to one row per config: its LAST-epoch loss and one-sided max viol.

    Returns columns: config, loss, max_viol, feasible.
    """
    c_cols = [c for c in df.columns if re.fullmatch(r"c_\d+", c)]
    if not c_cols:
        raise ValueError("no constraint columns (c_0, c_1, ...) found")

    # last epoch within each config
    df = df.sort_values("epoch")
    last = df.groupby("config").tail(tail).groupby("config").mean(numeric_only=True).reset_index()

    # one-sided max over signed constraints (matches g(x) <= eps enforcement)
    last["max_viol"] = last[c_cols].max(axis=1)
    last["feasible"] = last["max_viol"] <= bound

    if pinns:
        return last[["config", "loss", "max_viol", "test", "val", "feasible"]].reset_index(drop=True)  
    else:
        return last[["config", "loss", "max_viol", "feasible"]].reset_index(drop=True)


# ── aggregate one method across seeds ────────────────────────────────────────
def aggregate_method(spec: ExperimentSpec, method: str, split: str = "train", tail=10, 
                     last_epoch = True) -> Optional[pd.DataFrame]:
    """For one method, load all seeds, take each config's last epoch, then MEAN
    across seeds per config. last_epoch -> or running average

    Returns a per-config DataFrame with columns:
        config,
        loss_mean, loss_std,
        viol_mean, viol_std,
        feasible_mean,           # fraction of seeds in which this config was feasible
        feasible_all,            # True iff feasible in ALL seeds
        feasible_mean_viol       # feasibility judged on the MEAN violation (<= bound)
    or None if no seed files were found.
    """
    per_seed = []
    for seed in spec.seeds:
        if split == '':
            path = os.path.join(spec.seed_dir(seed), f"runs_{method}.csv")
        else: 
            path = os.path.join(spec.seed_dir(seed), f"runs_{method}_{split}.csv")
        if not os.path.exists(path):
            continue
        df = _read_runs_csv(path)
        if last_epoch:
            last = _last_epoch_per_config(df, spec.bound, spec.pinns, tail=tail)
        else: 
            last = _rolling_min_per_config(df, spec.bound, spec.pinns, tail=tail)
        last["seed"] = seed
        per_seed.append(last)

    if not per_seed:
        return None

    allseeds = pd.concat(per_seed, ignore_index=True)

    # mean/std across seeds, per config
    g = allseeds.groupby("config")

    if spec.pinns == True:
        out = pd.DataFrame({
            "test_mean": g["test"].mean(),
            "test_std":  g["test"].std(ddof=0),
            "val_mean": g["val"].mean(),
            "val_std": g["val"].std(ddof=0),
            "train_mean": g["loss"].mean(),
            "train_std":  g["loss"].std(ddof=0),
            "violation_constr_mean": g["max_viol"].mean(),          # fraction of seeds feasible
            "violation_constr_std":  g["max_viol"].std(ddof=0),           # feasible in every seed
        }).reset_index()
        # feasibility on the across-seed MEAN violation (the definition used for the CDF)
        out["feasible_mean_viol"] = out["violation_constr_mean"] <= spec.bound
        out["n_seeds_found"] = g["seed"].nunique().values
    else: 
        out = pd.DataFrame({
            "loss_mean": g["loss"].mean(),
            "loss_std":  g["loss"].std(ddof=0),
            "violation_constr_mean": g["max_viol"].mean(),
            "violation_constr_std":  g["max_viol"].std(ddof=0),
            "feasible_mean": g["feasible"].mean(),          # fraction of seeds feasible
            "feasible_all":  g["feasible"].all(),           # feasible in every seed
        }).reset_index()
        # feasibility on the across-seed MEAN violation (the definition used for the CDF)
        out["feasible_mean_viol"] = out["violation_constr_mean"] <= spec.bound
        out["n_seeds_found"] = g["seed"].nunique().values
    return out


# ── aggregate all methods for an experiment ──────────────────────────────────
DEFAULT_METHODS = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]


def aggregate_experiment(spec: ExperimentSpec, methods=DEFAULT_METHODS,
                         split: str = "train", tail=10, last_epoch=True) -> dict:
    """Returns {method: per-config DataFrame}. Methods with no files are skipped
    (with a printed note)."""
    result = {}
    for m in methods:
        agg = aggregate_method(spec, m, split=split, tail=tail, last_epoch=last_epoch)
        if agg is None:
            print(f"  [{spec.name}] {m}: no files found, skipping")
            continue
        n_feas_mean = int(agg["feasible_mean_viol"].sum())
        print(f"  [{spec.name}] {m}: {len(agg)} configs, "
              f"{n_feas_mean} feasible (mean viol <= {spec.bound}), "
              f"seeds found per config ~{int(agg['n_seeds_found'].median())}")
        result[m] = agg
    return result


# ── best config (val-selected) for the convergence plot ──────────────────────
def load_best_config(spec: ExperimentSpec, method: str, seed: int) -> Optional[dict]:
    """Load best_{method}.json written by the gridsearch (val-selected config)."""
    path = os.path.join(spec.seed_dir(seed), f"best_{method}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Example: edit to your actual experiment, then run to sanity-check aggregation.
    # spec = ExperimentSpec(
    #     name="E3",
    #     data="folktables",
    #     task="folktables_positive_rate_pair",
    #     bound=0.1,
    #     pinns=False,
    #     seeds=(0, 1, 2),
    #     results_root="results",
    # )
    # print(f"Aggregating {spec.name} from {spec.results_root}/ ...")
    # agg = aggregate_experiment(spec)
    # for method, df in agg.items():
    #     print(f"\n=== {method} ===")
    #     print(df.head().to_string(index=False))

    
    # Example: edit to your actual experiment, then run to sanity-check aggregation.
    spec = ExperimentSpec(
        name="E5",
        data="folktables",
        task="cifar10",
        bound=0.1,
        pinns=False,
        seeds=(0, 1, 2),
        results_root="results",
    )
    tail = 3

    print(f"Aggregating {spec.name} from {spec.results_root}/ ...")
    agg = aggregate_experiment(spec, split='train', tail=tail)
    for method, df in agg.items():
        print(f"\n=== {method} ===")
        print(df.head().to_string(index=False))