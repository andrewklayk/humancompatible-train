"""
plot_PINNs.py

Two-step PINN plotting:
  1) SELECT: aggregate across seeds, pick each method's best config by mean
     validation loss (among configs feasible on the mean violation; fall back to
     min val if none feasible).
  2) RELOAD + PLOT: for that config, reload the FULL per-epoch trajectory from
     each seed's runs_{method}.csv, stack across seeds (mean + std), and feed
     train-loss / test-loss / constraints into plot_losses_and_constraints_stochastic.

PINN column mapping (per the runs_{method}.csv schema):
    train loss  <- 'loss'  (PDE residual, the minimized objective)
    test  loss  <- 'test'  (relative L2 error vs analytic solution; PINN headline)
    constraints <- 'c_0' .. 'c_{m-1}'  (IC/BC residuals)
    test constraints: none (PINNs have no separate test-set constraints)
"""

import os
import re
import numpy as np
import pandas as pd

from aggregate_results import ExperimentSpec, aggregate_experiment, _read_runs_csv
from plotting import plot_losses_and_constraints_stochastic   # your existing module


# ── Step 1: best config per method (by mean val loss, feasible-preferred) ─────
def select_best_configs(spec: ExperimentSpec, methods, split="", best_validation_lastK=1):
    """Returns {method: best_config_index}. Selection on the across-seed MEAN
    validation loss, restricted to configs feasible on the mean violation; if no
    config is feasible, falls back to global min mean-val."""
    agg = aggregate_experiment(spec, methods=methods, split=split, tail=best_validation_lastK)
    best = {}
    for method, df in agg.items():
        pool = df
        # PINN aggregation names the val column 'val_mean'
        col = "val_mean" 
        best_row = pool.loc[pool[col].idxmin()]
        best[method] = int(best_row["config"])
        print(f"  [{spec.name}] {method}: best config {best[method]} "
              f"(mean val {best_row[col]:.4g}) ")
    return best


# ── Step 2: reload full trajectory of one config, stacked across seeds ────────
def _load_config_trajectory(spec: ExperimentSpec, method: str, config_idx: int,
                            split=""):
    """For the given config, load every seed's runs_{method}.csv, pull the full
    epoch series, and return per-seed stacked arrays:
        loss[seed, epoch], test[seed, epoch], cons[seed, m, epoch]
    Returns (loss, test, cons, m) or None if no files.
    """
    per_seed_loss, per_seed_test, per_seed_cons = [], [], []
    for seed in spec.seeds:
        fname = f"runs_{method}.csv" if split == "" else f"runs_{method}_{split}.csv"
        path = os.path.join(spec.seed_dir(seed), fname)
        if not os.path.exists(path):
            continue
        df = _read_runs_csv(path)
        sub = df[df["config"] == config_idx].sort_values("epoch")
        if sub.empty:
            continue
        c_cols = sorted([c for c in df.columns if re.fullmatch(r"c_\d+", c)],
                        key=lambda s: int(s.split("_")[1]))
        per_seed_loss.append(sub["loss"].to_numpy())
        per_seed_test.append(sub["test"].to_numpy())
        per_seed_cons.append(sub[c_cols].to_numpy().T)   # shape (m, epochs)

    if not per_seed_loss:
        return None
    # align epoch length defensively (seeds should match)
    L = min(len(x) for x in per_seed_loss)
    loss = np.stack([x[:L] for x in per_seed_loss])       # (seeds, L)
    test = np.stack([x[:L] for x in per_seed_test])       # (seeds, L)
    cons = np.stack([c[:, :L] for c in per_seed_cons])    # (seeds, m, L)
    m = cons.shape[1]
    return loss, test, cons, m


# ── assemble the lists the plotting function expects ─────────────────────────
def build_plot_inputs(spec: ExperimentSpec, methods, split="", best_validation_lastK=1):
    """Returns the argument lists for plot_losses_and_constraints_stochastic:
        train_losses (PDE residual), test_losses (solution error),
        train_constraints (m x epochs), and their stds; plus titles.
    Each list is per-method; arrays are mean / std across seeds."""
    best = select_best_configs(spec, methods, split=split, best_validation_lastK=best_validation_lastK)

    train_losses, train_losses_std = [], []
    test_losses, test_losses_std = [], []
    train_cons, train_cons_std = [], []
    titles = []

    for method in methods:
        if method not in best:
            continue
        traj = _load_config_trajectory(spec, method, best[method], split=split)
        if traj is None:
            print(f"  {method}: no trajectory for config {best[method]}, skipping")
            continue
        loss, test, cons, m = traj
        train_losses.append(loss.mean(0))
        train_losses_std.append(loss.std(0))
        test_losses.append(test.mean(0))
        test_losses_std.append(test.std(0))
        train_cons.append(cons.mean(0))          # (m, epochs)
        train_cons_std.append(cons.std(0))       # (m, epochs)
        titles.append(METHOD_LABELS.get(method, method))

    return dict(
        train_losses_list=train_losses,
        train_losses_std_list=train_losses_std,
        test_losses_list=test_losses,
        test_losses_std_list=test_losses_std,
        train_constraints_list=train_cons,
        train_constraints_std_list=train_cons_std,
        titles=titles,
    )


METHOD_LABELS = {
    "adam": "Adam", "pbm": "SPBM", "alm_proj": "SSL-ALM (proj.)",
    "alm_max": "SSL-ALM (max)", "ssg": "SSw",
}


def plot_PINNs(spec=None, methods=None, save_path=None, constraint_titles=None, best_validation_lastK=1):
    if spec is None:
        spec = ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1),
                              results_root="results")
    if methods is None:
        methods = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]

    inputs = build_plot_inputs(spec, methods, split="", best_validation_lastK=best_validation_lastK)
    if not inputs["train_losses_list"]:
        print("no data to plot")
        return

    plot_losses_and_constraints_stochastic(
        **inputs,
        constraint_thresholds=spec.bound,
        mode="train",            # train loss + test loss side by side
        separate_constraints=True,    # one row per constraint (IC, BC, ...)
        log_constraints=True,         # PINN residuals span orders of magnitude
        std_multiplier=1,
        save_path=save_path,
        constraint_titles=constraint_titles
    )


if __name__ == "__main__":

    spec = ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1),
                              results_root="results")
    constraint_titles = ["Initial Condition", "Boundary Condition"]

    best_validation_lastK = 5

    # takes the best validation loss config, then takes the solution from that config and plots the 
    # train / test loss and train constraints
    # the plot uses the weight (E1) plotting function
    plot_PINNs(spec = spec, save_path="./results/plots/pinn_burgers.png", 
               constraint_titles=constraint_titles, best_validation_lastK=best_validation_lastK)