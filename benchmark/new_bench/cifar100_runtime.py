"""No-tuning runtime/active-constraint sweep across algorithms on cifar100_loss.

Runs each algorithm's DEFAULT config (conf/algorithm/<algo>.yaml, no sweep/tuning)
for N_EPOCHS epochs, at each constraint-count in CONSTRAINT_GRID, repeated over
INIT_SEEDS, and logs per epoch:
  * epoch_time_sec -- wall-clock time of the training loop (forward + backward +
    step over every batch), same quantity as train.py's `time` column.
  * active_pct     -- for each batch, the fraction of the ENFORCED constraints
    with c_j > bound (violated/binding), averaged over all batches in the epoch.

CIFAR100's LossPairwise constraint produces m=9900 constraints (100x99 ordered
class pairs). To study how runtime/active_pct scale with the number of enforced
constraints, all but ``k`` of them are "muted": the full 9900-length constraint
vector is still computed every batch (it's a byproduct of the per-group loss
already needed for the forward pass), but only a fixed size-k subset is passed to
the algorithm's step (and sized into the dual optimizer) -- the rest are dropped
before backprop. The subset is a fixed (seed=0) random permutation of the 9900
indices, truncated to k, so grid points are nested (the k=100 subset is contained
in the k=1000 subset, etc.) and identical across algorithms at a given k.

``init_seed`` (model init + batch order) is repeated over INIT_SEEDS, same as the
rest of new_bench's `opt` approach -- there's no val/test split or folding here.

Usage:
    conda run -n hc-dev python cifar100_runtime.py
"""
import os
import time

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import algorithms as algo_mod
import data as data_mod
import tasks as tasks_mod
from train import calc_constraints

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
ALGOS = [
    "adam",
    "pbm",
    "alm_proj",
    "ssg"
]
CONSTRAINT_GRID = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9900]
N_EPOCHS = 3
INIT_SEEDS = [0, 1, 2, 3, 4]
MASK_SEED = 0  # fixed permutation seed so the muted subset is reused across algorithms/grid points


def _compose_cfg(algo):
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CONF_DIR, version_base=None)
    return compose(config_name="config", overrides=[
        f"algorithm={algo}", "data=cifar100", "task=cifar100_loss",
        f"n_epochs={N_EPOCHS}", "approach=opt", "verbose=false",
    ])


def _mute_mask(full_m, k, device):
    """Fixed size-k subset of range(full_m), nested across k (k=100 subset of k=1000, ...)."""
    perm = torch.randperm(full_m, device="cpu", generator=torch.Generator().manual_seed(MASK_SEED))
    return perm[:k].to(device)


def run_one(algo, n_constraints, init_seed, device):
    """Trains N_EPOCHS epochs for one (algo, n_constraints, init_seed); returns one
    record per epoch."""
    cfg = _compose_cfg(algo)

    torch.manual_seed(init_seed)
    bundle = data_mod.build_data(
        OmegaConf.to_container(cfg.data, resolve=True), batch_size=int(cfg.task.batch_size),
        device=device, cv_seed=int(cfg.cv_seed), n_folds=int(cfg.n_folds), fold=int(cfg.fold),
        init_seed=init_seed, approach="opt", opt_eval_size=int(cfg.opt_eval_size))
    task = tasks_mod.build_task(OmegaConf.to_container(cfg.task, resolve=True), bundle)
    k = min(n_constraints, task.m)
    idx = _mute_mask(task.m, k, device)

    torch.manual_seed(init_seed)  # reseed right before model init, as run.py does
    model = task.model_factory().to(device)
    algorithm = algo_mod.build_algorithm(
        cfg.algorithm, model, m=k, epoch_length=len(bundle.train_loader))

    bounds_full = torch.tensor([task.bound] * task.m, device=device)
    bounds = bounds_full[idx]
    c_to_eq = algorithm.constraints_to_eq

    records = []
    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        active_fracs = []
        start = time.perf_counter()
        for feats, sens, labels in bundle.train_loader:
            feats, sens, labels = feats.to(device), sens.to(device), labels.to(device)
            algorithm.zero_grad()
            out = model(feats)
            loss = task.loss_fn(out, labels)
            loss_for_c = loss if algorithm.passes_loss_to_constraints else None
            c_full, c_eq_full = calc_constraints(task.constraint_fn, bounds_full, task.fuse_loss_constraint,
                                                  c_to_eq, model, out, sens, labels, loss_for_c)
            c, c_eq = c_full[idx], c_eq_full[idx]  # drop the muted constraints before the step
            loss_mean = loss.mean() if loss.dim() > 0 else loss
            algorithm.step(loss_mean, c_eq)
            active_fracs.append((c.detach() > bounds).float().mean().item())
        epoch_time = time.perf_counter() - start
        active_pct = 100.0 * sum(active_fracs) / len(active_fracs)
        records.append({
            "algorithm": algo,
            "n_constraints": k,
            "init_seed": init_seed,
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "active_pct": active_pct,
            "active_count": active_pct / 100.0 * k,  # average number (not %) of enforced constraints active
        })
    return records


def _metric_table(df, metric, fmt):
    """Per-(algorithm, n_constraints) 'mean +/- std' of ``metric``, pooled over epochs
    and seeds; rows=n_constraints, columns=algorithm."""
    stats = df.groupby(["n_constraints", "algorithm"])[metric].agg(["mean", "std"])
    cells = stats.apply(lambda r: f"{r['mean']:{fmt}} $\\pm$ {r['std']:{fmt}}", axis=1).unstack("algorithm")
    return cells.reindex(index=CONSTRAINT_GRID, columns=[a for a in ALGOS if a in cells.columns])


def _write_latex_table(df, out_path):
    tables = [
        (_metric_table(df, "epoch_time_sec", ".2f"),
         "CIFAR100 per-epoch runtime (s), mean $\\pm$ std over epochs and init seeds.",
         "tab:cifar100_runtime"),
        (_metric_table(df, "active_count", ".1f"),
         "CIFAR100 average number of active (violated) constraints, mean $\\pm$ std over epochs and init seeds.",
         "tab:cifar100_active"),
    ]
    with open(out_path, "w") as f:
        for cells, caption, label in tables:
            f.write(cells.to_latex(escape=False, na_rep="--", caption=caption, label=label))
            f.write("\n")


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    results = []
    for algo in ALGOS:
        for k in CONSTRAINT_GRID:
            for seed in INIT_SEEDS:
                print(f"[runtime] running {algo} k={k} seed={seed} ...")
                recs = run_one(algo, k, seed, device)
                for r in recs:
                    print(f"[runtime] {algo} k={k} seed={seed} epoch={r['epoch']}: "
                          f"time={r['epoch_time_sec']:.1f}s  active={r['active_pct']:.2f}%")
                results.extend(recs)

    df = pd.DataFrame(results)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(out_dir, "cifar100_runtime_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[runtime] wrote {csv_path}")

    table_path = os.path.join(out_dir, "cifar100_runtime_table.tex")
    _write_latex_table(df, table_path)
    print(f"[runtime] wrote {table_path}")


if __name__ == "__main__":
    main()
