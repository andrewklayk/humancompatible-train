"""
Hyperparameter grid search for ALM and SPBM on the ACS fairness task.

ALM:  sweep dual lr  in ALM_LRS
SPBM: sweep (penalty_mult, gamma) in SPBM_GRID

For each configuration, train with a fixed eps and report the test-set
max fairness violation and BCE loss (mean ± std over SEEDS).
Results are saved per-method to benchmark_results/hparam_{name}.csv.
"""

import csv
import os
import itertools
import torch
import numpy as np
from pareto_fairness import (
    load_data,
    train_constrained,
    N_CONSTRAINTS,
    SEEDS, EPOCHS, BATCH_SIZE, LR_MODEL,
)
from humancompatible.train.dual_optim import ALM, PBM

# ── config ───────────────────────────────────────────────────────────────────
# Fixed constraint threshold for the grid search
SEARCH_EPS = 0.05

ALM_LRS   = [0.01, 0.02, 0.05]

SPBM_PENALTY_MULTS = [0.99, 0.999, 0.9999, 1.]
SPBM_GAMMAS        = [0.1, 0.25, 0.5]
SPBM_UPD           = ["dimin", "dimin_adapt"]

RESULTS_DIR = "benchmark_results"

# ── grid definitions ─────────────────────────────────────────────────────────
# Each entry: (config_label, make_dual_fn)
def alm_configs():
    for lr in ALM_LRS:
        yield {"lr": lr}, lambda eps, lr=lr: ALM(m=N_CONSTRAINTS, lr=lr, is_ineq=(eps > 0))

def spbm_configs():
    for pm, g, upd in itertools.product(SPBM_PENALTY_MULTS, SPBM_GAMMAS, SPBM_UPD):
        yield ({"penalty_mult": pm, "gamma": g, "update": upd},
               lambda eps, pm=pm, g=g: PBM(m=N_CONSTRAINTS, penalty_update=upd,
                                           penalty_mult=pm, gamma=g, penalty_range=(1., 2.)))

# ── sweep ────────────────────────────────────────────────────────────────────
def sweep_config(make_dual, eps, data):
    """data: flat 9-tuple (X_tr,y_tr,g_tr, X_val,y_val,g_val, X_te,y_te,g_te)"""
    X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te = data
    val_losses, val_vs, tr_losses, tr_vs = [], [], [], []
    for seed in SEEDS:
        *_, val_loss, val_viols, tr_loss, tr_viols, _ = train_constrained(
            make_dual, X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te, eps, seed
        )
        val_losses.append(val_loss); val_vs.append(val_viols)
        tr_losses.append(tr_loss);   tr_vs.append(tr_viols)

    val_arr = np.array(val_vs)  # (n_seeds, N_CONSTRAINTS)
    tr_arr  = np.array(tr_vs)
    return {
        "val_loss_mean":     np.mean(val_losses),           "val_loss_std":     np.std(val_losses),
        "val_fair_max_mean": np.mean(val_arr.max(axis=1)),  "val_fair_max_std": np.std(val_arr.max(axis=1)),
        "tr_loss_mean":      np.mean(tr_losses),            "tr_loss_std":      np.std(tr_losses),
        "tr_fair_max_mean":  np.mean(tr_arr.max(axis=1)),   "tr_fair_max_std":  np.std(tr_arr.max(axis=1)),
    }

# ── save / load ───────────────────────────────────────────────────────────────
def _csv_path(method_name):
    return os.path.join(RESULTS_DIR, f"hparam_{method_name}.csv")

def _save(rows, hparam_keys, method_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _csv_path(method_name)
    metric_keys = ["val_loss_mean", "val_loss_std", "val_fair_max_mean", "val_fair_max_std",
                   "tr_loss_mean",  "tr_loss_std",  "tr_fair_max_mean",  "tr_fair_max_std"]
    fieldnames = hparam_keys + metric_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved to {path}")

def _load(method_name):
    path = _csv_path(method_name)
    try:
        def _coerce(v):
            try: return float(v)
            except ValueError: return v

        with open(path, newline="") as f:
            rows = [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]
        print(f"  Loaded cached results from {path}")
        return rows
    except FileNotFoundError:
        return None

# ── run ───────────────────────────────────────────────────────────────────────
def run_grid(method_name, configs_fn, data):
    cached = _load(method_name)
    if cached is not None:
        return cached

    rows = []
    for hparams, make_dual in configs_fn():
        label = "  ".join(f"{k}={v}" for k, v in hparams.items())
        print(f"  [{method_name}] {label}")
        metrics = sweep_config(make_dual, SEARCH_EPS, data)
        rows.append({**hparams, **metrics})
        print(f"    val  loss={metrics['val_loss_mean']:.4f}  max_viol={metrics['val_fair_max_mean']:.4f}  |  "
              f"train loss={metrics['tr_loss_mean']:.4f}  max_viol={metrics['tr_fair_max_mean']:.4f}")

    hparam_keys = list(next(iter(configs_fn()))[0].keys())
    _save(rows, hparam_keys, method_name)
    return rows

def print_summary(method_name, rows, sort_by="val_fair_max_mean"):
    print(f"\n── {method_name} results (sorted by {sort_by}) ──")
    sorted_rows = sorted(rows, key=lambda r: r[sort_by])
    metric_keys = {"val_loss_mean", "val_loss_std", "val_fair_max_mean", "val_fair_max_std",
                   "tr_loss_mean",  "tr_loss_std",  "tr_fair_max_mean",  "tr_fair_max_std"}
    hparam_keys = [k for k in sorted_rows[0] if k not in metric_keys]
    header = "  ".join(f"{k:>12}" for k in hparam_keys)
    header += "  val_loss  val_max_viol  tr_loss  tr_max_viol"
    print(header)
    for r in sorted_rows:
        row_str = "  ".join(f"{r[k]:>12}" if isinstance(r[k], str) else f"{r[k]:>12.4g}" for k in hparam_keys)
        row_str += (f"  {r['val_loss_mean']:>8.4f}  {r['val_fair_max_mean']:>12.4f}"
                    f"  {r['tr_loss_mean']:>7.4f}  {r['tr_fair_max_mean']:>11.4f}")
        print(row_str)


if __name__ == "__main__":
    print("Loading data...")
    (X_tr, y_tr, g_tr), (X_val, y_val, g_val), (X_te, y_te, g_te) = load_data()
    data = X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te
    print(f"Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]} | eps={SEARCH_EPS}\n")

    print("Grid search: ALM")
    alm_rows  = run_grid("ALM",  alm_configs,  data)

    print("\nGrid search: SPBM")
    spbm_rows = run_grid("SPBM", spbm_configs, data)

    print_summary("ALM",  alm_rows)
    print_summary("SPBM", spbm_rows)
