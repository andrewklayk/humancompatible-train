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

running_average = False

METHOD_LABELS = {
    "adam": "Adam", "pbm": "SPBM", "alm_proj": "SSL-ALM (proj.)",
    "alm_max": "SSL-ALM (max)", "ssg": "SSw",
}

# ── Step 1: best config per method (by mean val loss, feasible-preferred) ─────
def select_best_configs(spec: ExperimentSpec, methods, split="", best_validation_lastK=1):
    """Returns {method: best_config_index}. Selection on the across-seed MEAN
    validation loss, restricted to configs feasible on the mean violation; if no
    config is feasible, falls back to global min mean-val."""
    agg = aggregate_experiment(spec, methods=methods, split=split, 
                               tail=best_validation_lastK, last_epoch=not running_average)
    best = {}
    for method, df in agg.items():
        pool = df[df["violation_constr_mean"] < 0.00011] if method != 'adam' else df # select feasible configs only
        if len(pool) == 0: # if empty, just select the lowest train loss
            pool = df
        col = "train_mean" 
        best_row = pool.loc[pool[col].idxmin()]
        best[method] = int(best_row["config"])
        print(f"  [{spec.name}] {method}: best config {best[method]} "
              f"(train loss {best_row[col]:.4g}) ")
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
        train_losses_list=np.array(train_losses),
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


def plot_PINNs(spec=None, methods=None, save_path=None, constraint_titles=None, 
               best_validation_lastK=1):
    if spec is None:
        spec = ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1),
                              results_root="results")
    if methods is None:
        methods = ["adam", "pbm", "alm_proj", "ssg"]

    inputs = build_plot_inputs(spec, methods, split="", best_validation_lastK=best_validation_lastK)
    if not inputs["test_losses_list"]:
        print("no data to plot")
        return

    inputs['train_losses_list'] += 1e-4  # avoid log(0) in plotting

    plot_losses_and_constraints_stochastic(
        **inputs,
        constraint_thresholds=spec.bound,
        mode="train",            # train loss + test loss side by side
        separate_constraints=True,    # one row per constraint (IC, BC, ...)
        log_constraints=True,         # PINN residuals span orders of magnitude
        std_multiplier=1,
        save_path=save_path,
        constraint_titles=constraint_titles,
        eval_points=10000,
        log_train_loss=True,
        log_test_loss=True
    )


    print(f"wrote {save_path}")

def plot_PINNs_single(specs, names, methods=None,
        save_path="./results/plots/pinns_convergence.pdf",
         best_validation_lastK=1):
   
    if methods is None:
        methods = ["adam","alm_proj", "ssg", "pbm"]

    # collect per-spec mean curves: {method: [curve_per_spec, ...]}
    per = {m: {"loss": [], "test": [], "cons": []} for m in methods}
    for name in names:

        inputs = build_plot_inputs(specs[name], methods, best_validation_lastK=best_validation_lastK)

        # per-spec baseline: best final value across methods (eps-floored)
        eps = 1e-4 
        base_loss = min(c[-1] for c in inputs["train_losses_list"]) + eps
        base_test = min(c[-1] for c in inputs["test_losses_list"]) + eps

        for i, title in enumerate(inputs["titles"]):
            m = next(k for k, v in METHOD_LABELS.items() if v == title)
            per[m]["loss"].append((inputs["train_losses_list"][i] + eps) / base_loss)
            per[m]["test"].append((inputs["test_losses_list"][i] + eps) / base_test)
            per[m]["cons"].append(inputs["train_constraints_list"][i].max(0))
    
    # mean/std across experiments
    train_l, train_s, test_l, test_s, cons_l, cons_s, titles = [], [], [], [], [], [], []
    for m in methods:
        if not per[m]["loss"]:
            continue
        L = min(len(c) for c in per[m]["loss"])
        stack = lambda key: np.stack([c[:L] for c in per[m][key]])
        loss, test, cons = stack("loss"), stack("test"), stack("cons")
        train_l.append(loss.mean(0)); train_s.append(loss.std(0))
        test_l.append(test.mean(0));  test_s.append(test.std(0))
        cons_l.append(cons.mean(0)[None, :]); cons_s.append(cons.std(0)[None, :])  # (1, epochs)
        titles.append(METHOD_LABELS.get(m, m))

    plot_losses_and_constraints_stochastic(
        train_losses_list=train_l, train_losses_std_list=train_s,
        test_losses_list=test_l, test_losses_std_list=test_s,
        train_constraints_list=cons_l, train_constraints_std_list=cons_s,
        titles=titles,
        constraint_thresholds=specs[names[0]].bound,
        mode="train", separate_constraints=True, log_constraints=True,
        std_multiplier=1, save_path=save_path,
        constraint_titles=["Max constraint violation"],
        eval_points=10000, log_train_loss=True, log_test_loss=True,
    )

    print(f"wrote {save_path}")

def print_table(names, methods):

    for name in names: 

        # for methods - store the tail of the losses and the tail of the max violation
        inputs = build_plot_inputs(spec, methods, split="", best_validation_lastK=best_validation_window)

        for idx, method in enumerate(methods): 

            # get the losses and the constraints
            loss = np.array(inputs['train_losses_list'][idx])
            constraints = np.array(inputs["train_constraints_list"][idx])
            loss_std = np.array(inputs["train_losses_std_list"][idx])
            constraints_std = np.array(inputs["train_constraints_std_list"][idx])

            # tail the loss and the constraints
            loss_tail = loss[-best_validation_window:].mean()
            loss_std_tail = loss_std[-best_validation_window:].mean()
            constraints_tail = constraints[:, -best_validation_window:].mean(axis=-1)
            constraints_std_tail = constraints_std[:, -best_validation_window:].mean(axis=-1)

            # compute the max violation
            worst_idx = constraints_tail.argmax()
            max_viol = max(0.0, constraints_tail[worst_idx])
            max_viol_std = constraints_std_tail[worst_idx]
            best_max_viol[name][method] = max_viol

            # store the values
            best_train_losses[name][method] = loss_tail
            best_train_losses_std[name][method] = loss_std_tail
            best_constraint_violations[name][method] = constraints_tail.mean()
            best_constraint_violations_std[name][method] = constraints_std_tail.mean()
            best_max_viol[name][method] = max_viol
            best_max_viol_std[name][method] = max_viol_std


    def fmt(x):
        return f"{x:.3f}"
    
    def rank_format(values_by_method, stds_by_method, methods, fmt=lambda x: f"{x:.3f}"):
        """Return {method: formatted_string} with best bold, second-best brown.
        Lower is better. Appends ± std in \\footnotesize."""
        ordered = sorted(methods, key=lambda m: values_by_method[m])
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        out = {}
        for m in methods:
            s = fmt(values_by_method[m])
            if m == best:
                cell = r"\textbf{" + s + "}"
            elif m == second and values_by_method[m] == values_by_method[best]:
                cell = r"\textbf{" + s + "}"
            elif m == second and values_by_method[m] != values_by_method[best]:
                cell = r"\textcolor{brown}{" + s + "}"
            else:
                cell = s
            cell += r" \footnotesize{$\pm$ " + fmt(stds_by_method[m]) + "}"
            out[m] = cell
        return out

    lines = [ r"\begin{table}[h]",
            r"\centering",
            r"\caption{Comparison of Adam, SSL-ALM, and SPBM on experiments \Exp{7} and \Exp{8}. We report the best test loss, together with the corresponding constraint violations (averaged over runs).}",
            r"\label{tab:best_results}",
        r"\begin{tabular}{l l c c c}",
        r"\toprule",
        r"Exp. & Method & Best loss & Mean constraint & Max constraint \\",
    ]

    for name in names:
        lines.append(r"\midrule")
        exp_id = name.split('E')[1]

        loss_cells = rank_format(best_train_losses[name],
                                best_train_losses_std[name], methods)
        mean_cells = rank_format(best_constraint_violations[name],
                                best_constraint_violations_std[name], methods)
        maxv_cells = rank_format(best_max_viol[name],
                                best_max_viol_std[name], methods)

        for i, method in enumerate(methods):
            multirow = (r"\multirow{" + str(len(methods)) + r"}{*}{\Exp{" + exp_id + r"}}"
                        if i == 0 else "")
            lines.append(
                f"{multirow} & {METHOD_LABELS[method]} & "
                f"{loss_cells[method]} & "
                f"{mean_cells[method]} & "
                f"{maxv_cells[method]} " + r"\\"
            )
        
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)
    print(table_str)

    # dump into a text file
    out = './results/tables/PINNS_latex_table.txt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(table_str)

if __name__ == "__main__":

    # True is a running window mean; False is a tail
    best_validation_window = 50

    names = ["E7", "E8", "E9"]
    specs = {
        "E7": ExperimentSpec(name="E7", data="helmholtz", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results2"),
        "E8": ExperimentSpec(name="E8", data="burgers", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results2"),
        "E9": ExperimentSpec(name="E9", data="klein_gordon", task="pinn",
                              bound=1e-4, pinns=True, seeds=(0, 1, 2, 3, 4),
                              results_root="results2"),
    }

    constraint_titles = ["Initial Condition", "Boundary Condition", "Boundary Condition 2"]

    # create an array for storing the best train loss and constraint violation for each method and experiment
    best_train_losses = {name: {} for name in names}
    best_constraint_violations = {name: {} for name in names}
    best_max_viol = {name: {} for name in names}
    best_train_losses_std = {name: {} for name in names}
    best_constraint_violations_std = {name: {} for name in names}
    best_max_viol_std = {name: {} for name in names}
    methods = ["adam", "alm_proj", "ssg", "pbm"]

    # iterate and plot all single plot for each experiment
    for name in names:
        spec = specs[name]
        # plot_PINNs(spec = spec, save_path=f"./results/plots/pinn_{spec.data}.pdf", 
        #         constraint_titles=constraint_titles, best_validation_lastK=best_validation_window)

    
    # print the latex table
    print_table(names, methods)

    # plot a single plot - combined all PINN experiments
    # plot_PINNs_single(specs, names, save_path=f"./results/plots/pinns_single.pdf", 
    #                     best_validation_lastK=best_validation_window)
