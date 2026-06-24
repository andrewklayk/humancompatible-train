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
def select_best_configs(spec: ExperimentSpec, methods, split="val", best_validation_lastK=1, threshold=0.1):
    """Returns {method: best_config_index}. Selection on the across-seed MEAN
    validation loss, restricted to configs feasible on the mean violation; if no
    config is feasible, falls back to global min mean-val."""
    agg = aggregate_experiment(spec, methods=methods, split=split, 
                               tail=best_validation_lastK, last_epoch=not running_average)
    best = {}
    for method, df in agg.items():
        pool = df[df["violation_constr_mean"] < threshold] if method != 'adam' else df
        if len(pool) == 0:
            pool = df
        col = "loss_mean" 
        best_row = pool.loc[pool[col].idxmin()]
        best[method] = int(best_row["config"])
        print(f"  [{spec.name}] {method}: best config {best[method]} "
              f"(mean val {best_row[col]:.4g}) ")
    return best


def _load_config_trajectory(spec, method, config_idx, split_train="train", split_test="test"):
    def load_one(split):
        per_seed = []
        for seed in spec.seeds:
            path = os.path.join(spec.seed_dir(seed), f"runs_{method}_{split}.csv")
            if not os.path.exists(path): continue
            df = _read_runs_csv(path)
            sub = df[df["config"]==config_idx].sort_values("epoch")
            if sub.empty: continue
            per_seed.append(sub)
        return per_seed
    tr, te = load_one(split_train), load_one(split_test)
    if not tr: return None
    c_cols = sorted([c for c in tr[0].columns if re.fullmatch(r"c_\d+", c)], key=lambda s:int(s.split("_")[1]))
    L = min(min(len(s) for s in tr), min(len(s) for s in te))
    loss = np.stack([s["loss"].to_numpy()[:L] for s in tr])      # train loss
    test = np.stack([s["loss"].to_numpy()[:L] for s in te])      # val loss (the "test" panel)
    cons_tr = np.stack([s[c_cols].to_numpy().T[:, :L] for s in tr]) # train constraints
    cons_te = np.stack([s[c_cols].to_numpy().T[:, :L] for s in te]) # train constraints
    return loss, test, cons_tr, cons_te, cons_tr.shape[1]

# ── assemble the lists the plotting function expects ─────────────────────────
def build_plot_inputs(spec: ExperimentSpec, methods, best_validation_lastK=1):
    """Returns the argument lists for plot_losses_and_constraints_stochastic:
        train_losses (PDE residual), test_losses (solution error),
        train_constraints (m x epochs), and their stds; plus titles.
    Each list is per-method; arrays are mean / std across seeds."""
    best = select_best_configs(spec, methods, split="val", best_validation_lastK=best_validation_lastK,
                               threshold=spec.bound)

    train_losses, train_losses_std = [], []
    test_losses, test_losses_std = [], []
    train_cons, train_cons_std = [], []
    test_cons, test_cons_std = [], []
    titles = []

    for method in methods:
        if method not in best:
            continue
        traj = _load_config_trajectory(spec, method, best[method], split_train="train", split_test="test")
        if traj is None:
            print(f"  {method}: no trajectory for config {best[method]}, skipping")
            continue
        loss, test, cons_tr, cons_te, m = traj
        train_losses.append(loss.mean(0))
        train_losses_std.append(loss.std(0))
        test_losses.append(test.mean(0))
        test_losses_std.append(test.std(0))
        train_cons.append(cons_tr.mean(0))          # (m, epochs)
        train_cons_std.append(cons_tr.std(0))       # (m, epochs)
        test_cons.append(cons_te.mean(0))          # (m, epochs)
        test_cons_std.append(cons_te.std(0))       # (m, epochs)
        titles.append(METHOD_LABELS.get(method, method))

    return dict(
        train_losses_list=train_losses,
        train_losses_std_list=train_losses_std,
        test_losses_list=test_losses,
        test_losses_std_list=test_losses_std,
        train_constraints_list=train_cons,
        train_constraints_std_list=train_cons_std,
        test_constraints_list=test_cons,
        test_constraints_std_list=test_cons_std,
        titles=titles,
    )


METHOD_LABELS = {
    "adam": "Adam", "pbm": "SPBM", "alm_proj": "SSL-ALM (proj.)",
    "alm_max": "SSL-ALM (max)", "ssg": "SSw",
}


def plot(spec=None, methods=None, save_path=None, constraint_titles=None, best_validation_lastK=1):
    if spec is None:
        spec = ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1),
                              results_root="results")
    if methods is None:
        methods = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]

    inputs = build_plot_inputs(spec, methods, best_validation_lastK=best_validation_lastK)
    if not inputs["train_losses_list"]:
        print("no data to plot")
        return

    if spec.name == "E1":
        inputs['test_constraints_list'] = None
        plot_losses_and_constraints_stochastic(
            **inputs,
            constraint_thresholds=spec.bound,
            mode="train",            # train loss + test loss side by side
            separate_constraints=False,    # one row per constraint (IC, BC, ...)
            log_constraints=False,         # PINN residuals span orders of magnitude
            std_multiplier=1,
            save_path=save_path,
            constraint_titles=constraint_titles
        )

    else: 

        plot_losses_and_constraints_stochastic(
            **inputs,
            constraint_thresholds=spec.bound,
            mode="train_test",            # train loss + test loss side by side
            separate_constraints=False,    # one row per constraint (IC, BC, ...)
            log_constraints=False,         # PINN residuals span orders of magnitude
            std_multiplier=1,
            save_path=save_path,
            constraint_titles=constraint_titles
        )


if __name__ == "__main__":

    # True is a running window mean; False is a tail
    running_average = True
    best_validation_window = 5

    # name = 'E1'
    # spec = ExperimentSpec(
    #     name=name,
    #     data="folktables",
    #     task="weight_norm",
    #     bound=2.0,
    #     pinns=False,
    #     seeds=(0, 1, 2, 3, 4),
    #     results_root="results",
    # )
    # constraint_titles = list(range(300))

    name = 'E2'
    spec = ExperimentSpec(
        name=name,
        data="folktables",
        task="equalized_odds_vec",
        bound=0.2,
        pinns=False,
        seeds=(0, 1, 2, 3, 4),
        results_root="results",
    )
    constraint_titles = list(range(300))


    # TODO: !!!! change val to test in the load config function

    # takes the best validation loss config, then takes the solution from that config and plots the 
    # train / test loss and train constraints
    # the plot uses the weight (E1) plotting function
    plot(spec = spec, save_path=f"./results/plots/{name}.png", 
               constraint_titles=constraint_titles, best_validation_lastK=best_validation_window)