"""plot_fair.py (new_bench) — per-method best-config trajectories.

Two steps, equivalent to ../../plotting/plot_fair.py:
  1) SELECT: read each method's winning config from select_best.py's best_*.json
     (select_best.py is the single selector -- no re-selection here).
  2) PLOT: for that config, load the seed-averaged per-epoch train/test loss and
     per-constraint curves from aggregate.py's selection/aggregated/, and feed them to
     the shared renderer plot_losses_and_constraints_stochastic (from ../../plotting/plotting.py).

The second column plots a chosen companion split alongside train: ``--companion``
(``val`` or ``test``, default ``test``).

Usage (run aggregate.py then select_best.py first):
    python plot_fair.py --task folktables_positive_rate_pair --data income \
        --bound 0.1 [--tol 1.0] [--companion val|test] [--out plots/fair.png]
"""
import argparse
import json
import os
import sys

import numpy as np

from prepare_results_plotting import ExperimentSpec, config_trajectory

# Shared renderer lives in the sibling ../../plotting package (pure matplotlib/numpy).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plotting"))
from plotting import plot_losses_and_constraints_stochastic  # noqa: E402

METHOD_LABELS = {
    "adam": "Adam", "pbm": "SPBM", "alm_proj": "SSL-ALM (proj.)",
    "alm_max": "SSL-ALM (max)", "ssg": "SSw",
}
plot_train_only = True


def read_best_configs(spec, methods, tol_mult=1.0):
    """{method: best_config_index}, read from select_best.py's best_*.json winners.

    No re-selection here -- select_best.py is the single selector. Looks up the
    per-(cell, slack) winner file for ``tol_mult``, falling back to the
    filter='none' pick (adam) or an untagged file. Methods with no winner at this
    slack (e.g. infeasible) are skipped."""
    sel_dir = os.path.dirname(os.path.abspath(spec.agg_root).rstrip("/"))  # selection/
    cell = f"{spec.task}_{spec.data}"
    best = {}
    for method in methods:
        base = os.path.join(sel_dir, f"best_{cell}_{method}")
        print(base)
        path = next((p for p in (f"{base}__tol{tol_mult:g}.json",
                                  f"{base}__none.json", f"{base}.json")
                     if os.path.exists(p)), None)
        if path is None:
            print(f"  [{spec.name}] {method}: no best_*.json at tol={tol_mult:g} "
                  f"(run select_best.py first), skipping")
            continue
        with open(path) as f:
            rec = json.load(f)
        best[method] = int(rec["config_index"])
        # print(f"  [{spec.name}] {method}: best config {best[method]} from "
        #       f"{os.path.basename(path)} (val loss {rec['val_loss_mean']:.4g}, "
        #       f"viol {rec['val_max_viol_mean']:.4g})")
    return best


def _load_config_trajectory(spec, method, config_idx, companion="test"):
    """Seed-averaged trajectories for one config, from the aggregated curves.

    Plots train + a ``companion`` split ('val' or 'test'). Returns
    (loss_m, loss_s, cons_tr_m, cons_tr_s, comp_m, comp_s, cons_co_m, cons_co_s);
    the comp_* / cons_co_* are None when the companion split is not stored (e.g.
    image tasks have no per-epoch val curve in some setups). Train and companion
    arrays are truncated to a common length."""
    tr = config_trajectory(spec, method, config_idx, "train")
    if tr is None:
        return None
    loss_m, loss_s, cons_tr_m, cons_tr_s = tr

    co = config_trajectory(spec, method, config_idx, companion)
    if co is not None:
        comp_m, comp_s, cons_co_m, cons_co_s = co
        L = min(len(loss_m), len(comp_m))
        loss_m, loss_s = loss_m[:L], loss_s[:L]
        comp_m, comp_s = comp_m[:L], comp_s[:L]
        if cons_tr_m is not None:
            cons_tr_m, cons_tr_s = cons_tr_m[:, :L], cons_tr_s[:, :L]
        if cons_co_m is not None:
            cons_co_m, cons_co_s = cons_co_m[:, :L], cons_co_s[:, :L]
    else:
        comp_m = comp_s = cons_co_m = cons_co_s = None
    return loss_m, loss_s, cons_tr_m, cons_tr_s, comp_m, comp_s, cons_co_m, cons_co_s


def build_plot_inputs(spec, methods, tol_mult=1.0, companion="test"):
    best = read_best_configs(spec, methods, tol_mult=tol_mult)
    keys = ["train_losses_list", "train_losses_std_list", "test_losses_list",
            "test_losses_std_list", "train_constraints_list", "train_constraints_std_list",
            "test_constraints_list", "test_constraints_std_list", "titles"]
    acc = {k: [] for k in keys}
    any_test = False
    for method in methods:
        if method not in best:
            continue
        traj = _load_config_trajectory(spec, method, best[method], companion=companion)
        if traj is None:
            print(f"  {method}: no trajectory for config {best[method]}, skipping")
            continue
        loss_m, loss_s, cons_tr_m, cons_tr_s, comp_m, comp_s, cons_co_m, cons_co_s = traj
        acc["train_losses_list"].append(loss_m)
        acc["train_losses_std_list"].append(loss_s)
        acc["train_constraints_list"].append(cons_tr_m)
        acc["train_constraints_std_list"].append(cons_tr_s)
        # The renderer's "test" slot carries the chosen companion split (val or test).
        if comp_m is not None:
            any_test = True
            acc["test_losses_list"].append(comp_m)
            acc["test_losses_std_list"].append(comp_s)
            acc["test_constraints_list"].append(cons_co_m)
            acc["test_constraints_std_list"].append(cons_co_s)
        acc["titles"].append(METHOD_LABELS.get(method, method))
    if not any_test:  # no companion panel -> let the renderer draw train only
        for k in ["test_losses_list", "test_losses_std_list",
                  "test_constraints_list", "test_constraints_std_list"]:
            acc[k] = None
    return acc, any_test


def plot(spec, methods=None, save_path=None, tol_mult=1.0, constraint_titles=None,
         companion="test"):
    if methods is None:
        methods = ["adam", "pbm", "alm_proj", "ssg"]
        # methods = ["pbm", "alm_proj"]
    inputs, any_comp = build_plot_inputs(spec, methods, tol_mult=tol_mult, companion=companion)
    if not inputs["train_losses_list"]:
        print("no data to plot")
        return
    
    if plot_train_only:

        inputs["test_losses_list"] = None
        inputs["test_losses_std_list"] = None
        inputs["test_constraints_list"] = None
        inputs["test_constraints_std_list"] = None
        plot_losses_and_constraints_stochastic(
            **inputs,
            constraint_thresholds=spec.bound,
            mode="train",
            separate_constraints=False,
            log_constraints=False,
            std_multiplier=1,
            save_path=save_path,
            constraint_titles=constraint_titles,
        )
    else: 
        plot_losses_and_constraints_stochastic(
            **inputs,
            constraint_thresholds=spec.bound,
            mode="train_test" if any_comp else "train",
            separate_constraints=False,
            log_constraints=False,
            std_multiplier=1,
            save_path=save_path,
            constraint_titles=constraint_titles,
        )

    print(f"\nwrote {save_path} (train + {companion})")


if __name__ == "__main__":

    # ap = argparse.ArgumentParser()
    # ap.add_argument("--agg", default="../selection/",
    #                 help="dir of aggregate.py's per-cell aggregates (curves); select_best.py's "
    #                      "best_*.json winners are read from its parent. Run aggregate.py then "
    #                      "select_best.py first.")
    # ap.add_argument("--task", default="folktables_positive_rate_pair")
    # ap.add_argument("--data", default="income")
    # ap.add_argument("--bound", type=float, default=0.1)
    # ap.add_argument("--tol", type=float, default=1.0,
    #                 help="which select_best.py feasibility-slack winner to plot "
    #                      "(matches a --tols value, e.g. 1.0, 1.1, 1.25)")
    # ap.add_argument("--companion", default="train", choices=["train", "val", "test"],
    #                 help="which split to plot alongside train (second column)")
    # ap.add_argument("--out", default="../results/plots/fair.png")
    # args = ap.parse_args()
    # os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
 
    # all possible experiments
    experiments = [ 'weight_norm',
                    'folktables_positive_rate_vec',
                    'folktables_positive_rate_pair', 
                    'dutch_positive_rate_pair']

    data_map = {    "weight_norm": "income_norm",
                    "folktables_positive_rate_vec": "income", 
                    "folktables_positive_rate_pair": "income",
                    "dutch_positive_rate_pair": "dutch"
    }
    bounds_map = {  "weight_norm": 2.0,
                    "folktables_positive_rate_vec": 0.2, 
                    "folktables_positive_rate_pair": 0.1,
                    "dutch_positive_rate_pair": 0.1
    }

    # map to the E 
    mapping_name = {"weight_norm": "E1",
                    "folktables_positive_rate_vec": "E2", 
                    "folktables_positive_rate_pair": "E3",
                    "dutch_positive_rate_pair": "E4"}

    # define output folder
    out = "../../results/plots/"
    agg = "../selection/aggregated/"
    
    os.makedirs(out, exist_ok=True)

    specs = []

    # loop over all experiments and create the experiments
    for experiment in experiments: 
        
        # load the details about the experiment
        task = experiment
        data = data_map[experiment]
        bound = bounds_map[experiment]

        spec = ExperimentSpec(name=task , task=task, data=data,
                        bound=bound, agg_root=agg)

        specs.append(spec)

    # plot each experiment separately
    for i, experiment in enumerate(experiments):

        plot(specs[i], save_path=out + f"{mapping_name[experiment]}.pdf", tol_mult=1, companion="train",
            constraint_titles=list(range(300)))
