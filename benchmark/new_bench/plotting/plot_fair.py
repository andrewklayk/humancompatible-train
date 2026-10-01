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
from prepare_results_plotting import ExperimentSpec, config_trajectory, acc_trajectory, best_config_in

# Shared renderer lives in the sibling ../../plotting package (pure matplotlib/numpy).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plotting"))
from plotting import plot_losses_and_constraints_stochastic  # noqa: E402

METHOD_LABELS = {
    "adam": "Adam",
    "pbm": "SPBM",
    "pbm_mu0": r"SPBM (\mu=0)",
    "pbm_gamma0": r"SPBM (\gamma=0)",
    "pbm_kappa0": r"SPBM (\kappa=0)",
    # "pbm_dual": r"SPBM (\kappa=0)"
    "alm_proj": "SSL-ALM (proj.)",
    "ssg": "SSw",
}
plot_train_only = True
tail = 5


def read_best_configs(spec, methods, tol_mult=1.0):
    """{method: (source_algorithm, best_config_index)}, read from select_best.py's
    best_*.json winners.

    No re-selection here -- select_best.py is the single selector. Looks up the
    per-(cell, slack) winner file for ``tol_mult``, falling back to the
    filter='none' pick (adam) or an untagged file. Methods with no winner at this
    slack (e.g. infeasible) are skipped.

    ``source_algorithm`` is the raw cell the winning config actually lives in --
    for a pooled cell (select_best.py's --pool_subtypes) this may differ from
    ``method`` itself (e.g. method="alm_proj" but the winner came from
    "alm_proj_fix"), and config_index is only unique WITHIN that source cell's
    own aggregated file, so callers must use source_algorithm, not method, to
    locate the winning trajectory. Falls back to ``method`` for best_*.json
    files written before that field existed.
    """
    sel_dir = os.path.dirname(os.path.abspath(spec.agg_root).rstrip("/"))  # selection/
    cell = f"{spec.task}_{spec.data}"
    best = {}
    for method in methods:
        base = os.path.join(sel_dir, f"best_{cell}_{method}")
        print(base)
        path = next((p for p in (f"{base}__tol{tol_mult:g}.json", f"{base}.json")
                     if os.path.exists(p)), None)
        if path is None:
            print(f"  [{spec.name}] {method}: no best_*.json at tol={tol_mult:g} "
                  f"(run select_best.py first), skipping")
            continue
        with open(path) as f:
            rec = json.load(f)
        best[method] = (rec.get("source_algorithm", method), int(rec["config_index"]))

    return best


def _load_config_trajectory(spec, method, config_idx, companion="test"):
    """Seed-averaged trajectories for one config, from the aggregated curves.

    Plots train + a ``companion`` split ('val' or 'test'). Returns
    (loss_m, loss_s, cons_tr_m, cons_tr_s, comp_m, comp_s, cons_co_m, cons_co_s);
    the comp_* / cons_co_* are None when the companion split is not stored (e.g.
    image tasks have no per-epoch val curve in some setups). Train and companion
    arrays are truncated to a common length."""
    tr = config_trajectory(spec, method, config_idx, "opt")
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


def build_plot_inputs(spec, methods, tol_mult=1.0, companion="test", with_acc=False,
                       split_by=None, pool_subtypes=False):
    """split_by: optional {method: (dotted_hparam, value)} -- instead of that method's
    single select_best.py winner, plots two independently-best-selected entries: configs
    matching hparam==value, and the complement ("other" values). See
    prepare_results_plotting.best_config_in for the per-half selection rule.

    pool_subtypes: forwarded to best_config_in for methods in split_by, so the
    split re-selection also pools subtype cells (e.g. alm_proj_fix into alm_proj)
    the same way select_best.py's --pool_subtypes does for the non-split_by
    winner -- False (default)/True/a collection of subtype names."""
    split_by = split_by or {}
    best = read_best_configs(spec, [m for m in methods if m not in split_by], tol_mult=tol_mult)
    keys = ["train_losses_list", "train_losses_std_list", "test_losses_list",
            "test_losses_std_list", "train_constraints_list", "train_constraints_std_list",
            "test_constraints_list", "test_constraints_std_list", "titles"]
    if with_acc:  # per-class accuracy row (image tasks only), aligned with `titles`
        keys += ["train_acc_list", "train_acc_std_list"]
    acc = {k: [] for k in keys}
    any_test = False

    def add_entry(method, config_idx, title):
        nonlocal any_test
        traj = _load_config_trajectory(spec, method, config_idx, companion=companion)
        if traj is None:
            print(f"  {method}: no trajectory for config {config_idx}, skipping")
            return
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
        if with_acc:  # train per-class accuracy [K, L] (None if not aggregated)
            at = acc_trajectory(spec, method, config_idx, "train")
            acc["train_acc_list"].append(at[0] if at is not None else None)
            acc["train_acc_std_list"].append(at[1] if at is not None else None)
        acc["titles"].append(title)

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        if method in split_by:
            hkey, hval = split_by[method]
            for where, title in [({hkey: hval}, f"{label} ({hkey}={hval})"),
                                  ({hkey: lambda v, hval=hval: v != hval}, f"{label} (other {hkey})")]:
                source_method, cfg_idx = best_config_in(spec, method, where, pool_subtypes=pool_subtypes)
                if cfg_idx is None:
                    print(f"  [{spec.name}] {method} split {hkey}: no matching config, skipping")
                    continue
                add_entry(source_method, cfg_idx, title)
        else:
            if method not in best:
                continue
            source_method, config_idx = best[method]
            add_entry(source_method, config_idx, label)
    if not any_test:  # no companion panel -> let the renderer draw train only
        for k in ["test_losses_list", "test_losses_std_list",
                  "test_constraints_list", "test_constraints_std_list"]:
            acc[k] = None
    return acc, any_test


def plot(spec, methods=None, save_path=None, tol_mult=1.0, constraint_titles=None,
         companion="test", split_by=None, pool_subtypes=False):
    if methods is None:
        methods = ["adam","alm_proj", "ssg", "pbm"]
    # Per-class accuracy row only for the image tasks (they store per-class acc).
    with_acc = spec.data in ("cifar10", "cifar100")
    inputs, any_comp = build_plot_inputs(spec, methods, tol_mult=tol_mult,
                                         companion=companion, with_acc=with_acc,
                                         split_by=split_by, pool_subtypes=pool_subtypes)
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



def print_table(specs, methods, tol=1.1, split_by=None, pool_subtypes=False):
    """Rows are keyed by build_plot_inputs's `titles` (not the raw `methods` list) so a
    split method (see `split_by`) contributes its own row per half, and a method with
    no available config for a given spec is simply absent from that spec's rows rather
    than misaligning the rest via positional indexing."""

    # create an array for storing the best train loss and constraint violation for each method and experiment
    best_train_losses = {spec.task: {} for spec in specs}
    best_constraint_violations = {spec.task: {} for spec in specs}
    best_max_viol = {spec.task: {} for spec in specs}
    best_train_losses_std = {spec.task: {} for spec in specs}
    best_constraint_violations_std = {spec.task: {} for spec in specs}
    best_max_viol_std = {spec.task: {} for spec in specs}
    row_order = []  # preserves first-seen order of titles across specs

    for spec in specs:

        # for methods - store the tail of the losses and the tail of the max violation
        inputs, _ = build_plot_inputs(spec, methods, tol_mult=tol, companion="train",
                                      split_by=split_by, pool_subtypes=pool_subtypes)

        for idx, title in enumerate(inputs["titles"]):
            if title not in row_order:
                row_order.append(title)

            # get the losses and the constraints
            loss = np.array(inputs['train_losses_list'][idx])
            constraints = np.array(inputs["train_constraints_list"][idx])
            loss_std = np.array(inputs["train_losses_std_list"][idx])
            constraints_std = np.array(inputs["train_constraints_std_list"][idx])

            # tail the loss and the constraints
            loss_tail = loss[-tail:].mean()
            loss_std_tail = loss_std[-tail:].mean()
            constraints_tail = constraints[:, -tail:].mean(axis=-1)
            constraints_std_tail = constraints_std[:, -tail:].mean(axis=-1)

            # compute the max violation
            worst_idx = constraints_tail.argmax()
            max_viol = max(0.0, constraints_tail[worst_idx] - spec.bound)
            max_viol_std = constraints_std_tail[worst_idx]

            # store the values
            best_train_losses[spec.task][title] = loss_tail
            best_train_losses_std[spec.task][title] = loss_std_tail
            best_constraint_violations[spec.task][title] = constraints_tail.mean()
            best_constraint_violations_std[spec.task][title] = constraints_std_tail.mean()
            best_max_viol[spec.task][title] = max_viol
            best_max_viol_std[spec.task][title] = max_viol_std

    def rank_format(values_by_method, stds_by_method, methods,
                    precision=4, mark=True, tol=1e-5):
        """{method: formatted cell}, best bold, second-best brown (lower is better).
        Appends ± std. mark=False disables highlighting."""

        fmt = lambda x: f"{x:.{precision}f}"

        # round once, up front — everything downstream uses rounded values
        vals = {m: round(values_by_method[m], precision) for m in methods}

        def cell(m, wrap):
            body = wrap(fmt(vals[m])) if wrap else fmt(vals[m])
            return body + r" \footnotesize{$\pm$ " + fmt(stds_by_method[m]) + "}"

        if not mark:
            return {m: cell(m, None) for m in methods}


        ordered = sorted(methods, key=lambda m: vals[m])
        best_val = vals[ordered[0]]
        second_val = vals[ordered[1]] if len(ordered) > 1 else None

        bold  = lambda s: r"\textbf{" + s + "}"
        brown = lambda s: r"\textcolor{brown}{" + s + "}"

        out = {}
        for m in methods:
            v = vals[m]
            if abs(v - best_val) < tol:
                out[m] = cell(m, bold)
            elif second_val is not None and abs(v - second_val) < tol:
                out[m] = cell(m, brown)
            else:
                out[m] = cell(m, None)
        return out

    lines = [ r"\begin{table}[h]",
            r"\centering",
            r"\caption{Comparison of Adam, SSL-ALM, and SPBM on experiments \Exp{7} and \Exp{8}. We report the best test loss, together with the corresponding constraint violations (averaged over runs).}",
            r"\label{tab:best_results}",
        r"\begin{tabular}{l l c c c}",
        r"\toprule",
        r"Exp. & Method & Best loss & Max constraint viol. & Mean constraint \\",
    ]

    for spec in specs:
        lines.append(r"\midrule")
        exp_id = mapping_name[spec.task].split('E')[1]

        rows = [t for t in row_order if t in best_train_losses[spec.task]]

        loss_cells = rank_format(best_train_losses[spec.task],
                                best_train_losses_std[spec.task], rows,
                                precision=3)
        mean_cells = rank_format(best_constraint_violations[spec.task],
                                best_constraint_violations_std[spec.task], rows,
                                precision=4, mark=False)
        maxv_cells = rank_format(best_max_viol[spec.task],
                                best_max_viol_std[spec.task], rows,
                                precision=4)

        for i, title in enumerate(rows):
            multirow = (r"\multirow{" + str(len(rows)) + r"}{*}{\Exp{" + exp_id + r"}}"
                        if i == 0 else "")
            lines.append(
                f"{multirow} & {title} & "
                f"{loss_cells[title]} & "
                f"{maxv_cells[title]} & "
                f"{mean_cells[title]} " + r"\\"
            )
        
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table_str = "\n".join(lines)
    print(table_str)

    # dump into a text file
    out = '/mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/results/tables/FAIR_latex_table.txt'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(table_str)

if __name__ == "__main__":
    
    # all possible experiments
    experiments = [ 
        # 'weight_norm',
        'folktables_positive_rate_pair', 
        'dutch_positive_rate_pair',
        # 'cifar10_loss',
        # "cifar100_loss"
    ]


    data_map = {    "weight_norm": "income_norm",
                    "folktables_positive_rate_vec": "income", 
                    "folktables_positive_rate_pair": "income",
                    "dutch_positive_rate_pair": "dutch",
                    "cifar10_loss": "cifar10",
                    "cifar100_loss": "cifar100"
    }
    bounds_map = {  "weight_norm": 2.0,
                    "folktables_positive_rate_vec": 0.2, 
                    "folktables_positive_rate_pair": 0.1,
                    "dutch_positive_rate_pair": 0.1,
                     'cifar10_loss': 0.1,
                     'cifar100_loss': 0.1
    }

    # map to the E 
    mapping_name = {"weight_norm": "E1",
                    "folktables_positive_rate_pair": "E2",
                    "dutch_positive_rate_pair": "E3",
                     'cifar10_loss': "E4",
                     'cifar100_loss': "E5",
                     }

    # define output folder
    # out = "../../results/plots/"
    out = "/mnt/personal/kliacand/humancompatible-train/benchmark/new_bench/plotting/plots/"
    # agg = "../selection/aggregated/"
    agg = "/mnt/data/optimization/current/best_noablation/aggregated/"
    
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

    tol = 1.1

    methods = [
        "adam",
        "pbm",
        "alm_proj",
        "ssg",
    ]
    split_by = {}


    # methods = [
    #     # "alm_proj",
    #     "pbm",
    #     "pbm_gamma0",
    #     "pbm_kappa0",
    #     "pbm_mu0"
    # ]
    # split_by = {
    #     "pbm": ("dual.penalty_update", "alm"),
    # }


    POOL_SUBTYPES = False
    # plot each experiment separately
    for i, experiment in enumerate(experiments):

        plot(specs[i], save_path=out + f"{mapping_name[experiment]}.pdf", tol_mult=tol, companion="train",
            constraint_titles=list(range(3000000)), methods=methods, split_by=split_by, pool_subtypes = POOL_SUBTYPES)

    print_table(specs, methods, split_by=split_by, tol=tol, pool_subtypes = POOL_SUBTYPES)

    