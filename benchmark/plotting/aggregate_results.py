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


def _last_epoch_per_config(df: pd.DataFrame, bound: float,  pinns: bool) -> pd.DataFrame:
    """Collapse to one row per config: its LAST-epoch loss and one-sided max viol.

    Returns columns: config, loss, max_viol, feasible.
    """
    c_cols = [c for c in df.columns if re.fullmatch(r"c_\d+", c)]
    if not c_cols:
        raise ValueError("no constraint columns (c_0, c_1, ...) found")

    # last epoch within each config
    idx = df.groupby("config")["epoch"].idxmax()
    last = df.loc[idx].copy()

    # one-sided max over signed constraints (matches g(x) <= eps enforcement)
    last["max_viol"] = last[c_cols].max(axis=1)
    last["feasible"] = last["max_viol"] <= bound

    if pinns:
        return last[["config", "loss", "max_viol", "test", "val", "feasible"]].reset_index(drop=True)  
    else:
        return last[["config", "loss", "max_viol", "feasible"]].reset_index(drop=True)


# ── aggregate one method across seeds ────────────────────────────────────────
def aggregate_method(spec: ExperimentSpec, method: str, split: str = "train"
                     ) -> Optional[pd.DataFrame]:
    """For one method, load all seeds, take each config's last epoch, then MEAN
    across seeds per config.

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
        last = _last_epoch_per_config(df, spec.bound, spec.pinns)
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
                         split: str = "train") -> dict:
    """Returns {method: per-config DataFrame}. Methods with no files are skipped
    (with a printed note)."""
    result = {}
    for m in methods:
        agg = aggregate_method(spec, m, split=split)
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


# ── lastK-based aggregation and best-config selection ────────────────────────

def aggregate_method_lastK(spec: ExperimentSpec, method: str) -> Optional[pd.DataFrame]:
    """Load summary_{method}_val.csv for every seed (written by compute_lastK_summary
    in run_gridsearch.py) and aggregate across seeds per config.

    Each summary CSV has columns: config, loss_lastK, max_viol_lastK, where
    each value is already the mean over the last K training epochs for that config
    and seed.  Here we mean/std those across seeds.

    Returns a DataFrame with columns:
        config,
        loss_mean, loss_std,
        viol_mean, viol_std,
        feasible_mean,      # fraction of seeds where mean-last-K viol <= bound
        feasible_all,       # True iff feasible in every seed
        feasible_mean_viol, # feasibility judged on the cross-seed mean violation
        n_seeds_found
    or None if no files were found.
    """
    per_seed = []
    for seed in spec.seeds:
        path = os.path.join(spec.seed_dir(seed), f"summary_{method}_val.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)  # columns: config, loss_lastK, max_viol_lastK
        df["seed"] = seed
        df["feasible"] = df["max_viol_lastK"] <= spec.bound
        per_seed.append(df)

    if not per_seed:
        return None

    allseeds = pd.concat(per_seed, ignore_index=True)
    g = allseeds.groupby("config")

    out = pd.DataFrame({
        "loss_mean":  g["loss_lastK"].mean(),
        "loss_std":   g["loss_lastK"].std(ddof=0),
        "viol_mean":  g["max_viol_lastK"].mean(),
        "viol_std":   g["max_viol_lastK"].std(ddof=0),
        "feasible_mean": g["feasible"].mean(),
        "feasible_all":  g["feasible"].all(),
        "n_seeds_found": g["seed"].nunique(),
    }).reset_index()

    out["feasible_mean_viol"] = out["viol_mean"] <= spec.bound
    return out


def select_best_config_lastK(agg: pd.DataFrame) -> Optional[pd.Series]:
    """Select the config with the lowest cross-seed mean loss among configs whose
    cross-seed mean violation is feasible.  Falls back to the least-violating config
    if none are feasible.

    Returns a single-row Series (or None if agg is empty).
    """
    if agg is None or agg.empty:
        return None
    feasible = agg[agg["feasible_mean_viol"]]
    pool = feasible if not feasible.empty else agg
    return pool.loc[pool["loss_mean"].idxmin()]


def aggregate_experiment_lastK(spec: ExperimentSpec,
                                methods=DEFAULT_METHODS) -> dict:
    """Returns {method: (per-config DataFrame, best-config Series)}.
    Methods with no summary files are skipped."""
    result = {}
    for m in methods:
        agg = aggregate_method_lastK(spec, m)
        if agg is None:
            print(f"  [{spec.name}] {m}: no summary files found, skipping")
            continue
        best = select_best_config_lastK(agg)
        n_feas = int(agg["feasible_mean_viol"].sum())
        print(f"  [{spec.name}] {m}: {len(agg)} configs, "
              f"{n_feas} feasible (mean viol <= {spec.bound}), "
              f"best config={int(best['config'])} "
              f"loss={best['loss_mean']:.4f} viol={best['viol_mean']:.4f}")
        result[m] = (agg, best)
    return result


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
        name="E8",
        data="burgers",
        task="pinn",
        bound=0.0001,
        pinns=True,
        seeds=(0, 1),
        results_root="results",
    )
    print(f"Aggregating {spec.name} from {spec.results_root}/ ...")
    agg = aggregate_experiment(spec, split='')
    for method, df in agg.items():
        print(f"\n=== {method} ===")
        print(df.head().to_string(index=False))