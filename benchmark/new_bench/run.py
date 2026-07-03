"""Single-config benchmark entrypoint.

One invocation = ONE (data, task, algorithm, hyperparameter combo, seed). Hydra
multirun (``-m``) produces the cartesian product, each as its own job/dir. This
script does NO grid search and NO best-param selection -- it only trains one
config and writes its raw per-epoch histories. Selection is a separate step
(``select_best.py``) over the written results.

Examples
--------
Single run:
    python run.py data=income task=folktables_positive_rate_pair algorithm=pbm

Manual grid (Hydra basic sweeper) over a stored per-algorithm grid, across the
cross-validation axes:
    python run.py -m +sweep=pbm data=income task=folktables_positive_rate_pair \
        fold=0,1,2,3,4 init_seed=0,1,2
"""
import json
import os

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import algorithms as algo_mod
import data as data_mod
import tasks as tasks_mod
from train import train


def _write_history(history, path):
    if not history:
        return
    pd.DataFrame(history).set_index("epoch").to_csv(path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    init_seed = int(cfg.init_seed)
    cv_seed = int(cfg.cv_seed)
    n_folds = int(cfg.n_folds)
    fold = int(cfg.fold)
    approach = str(cfg.get("approach", "ml"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.manual_seed(init_seed)

    # --- data ---
    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    bundle = data_mod.build_data(
        cfg_data, batch_size=int(cfg.task.batch_size), device=device,
        cv_seed=cv_seed, n_folds=n_folds, fold=fold, init_seed=init_seed,
        approach=approach, opt_eval_size=int(cfg.get("opt_eval_size", 10000)))

    # --- task (model factory, loss, constraint, m, bound) ---
    task = tasks_mod.build_task(OmegaConf.to_container(cfg.task, resolve=True), bundle)

    # --- model + algorithm ---
    # Reseed (init_seed) right before model init so weight init is reproducible and
    # independent of data-loading RNG consumption.
    torch.manual_seed(init_seed)
    model = task.model_factory()
    algorithm = algo_mod.build_algorithm(
        cfg.algorithm, model, m=task.m, epoch_length=len(bundle.train_loader)
    )

    print(f"[run] task={cfg.task.name} algo={algorithm.name} approach={approach} "
          f"fold={fold}/{n_folds} init_seed={init_seed} "
          f"m={task.m} bound={task.bound} device={device}")

    # --- train ---
    h_train, h_val, h_test, h_opt = train(model, algorithm, task, bundle,
                                          n_epochs=int(cfg.n_epochs), device=device,
                                          approach=approach, verbose=cfg.verbose)

    # --- write raw results to this job's output dir ---
    hc = HydraConfig.get()
    out_dir = hc.runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)
    _write_history(h_train, os.path.join(out_dir, "train.csv"))
    _write_history(h_val, os.path.join(out_dir, "val.csv"))
    _write_history(h_test, os.path.join(out_dir, "test.csv"))
    _write_history(h_opt, os.path.join(out_dir, "opt.csv"))  # KKT metrics (opt mode)

    # Config-group choices identify the "benchmark cell" (data+task+algorithm) that
    # the selection step groups by; the same hyperparameters under different seeds
    # share these. Fall back to the .name fields if choices are unavailable.
    choices = hc.runtime.choices
    def _choice(key, default):
        try:
            return choices[key]
        except Exception:
            return default

    # Compact metadata for the selection step (avoids re-parsing Hydra configs).
    # fold / init_seed are NOT part of the config signature (they are the CV and
    # init axes select_best aggregates over); hyperparameters excludes them.
    meta = {
        "task": _choice("task", cfg.task.name),
        "data": _choice("data", cfg.data.get("name")),
        "algorithm": _choice("algorithm", algorithm.name),
        "approach": approach,
        "init_seed": init_seed,
        "fold": fold,
        "n_folds": n_folds,
        "cv_seed": cv_seed,
        "m": task.m,
        "bound": task.bound,
        "select_filter": algorithm.select_filter,
        "hyperparameters": OmegaConf.to_container(cfg.algorithm, resolve=True),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"[run] wrote results to {out_dir}")

    # Selection is post-hoc (select_best.py); the basic sweeper ignores the return
    # value. Returned only so a sweeper that expects a float objective stays happy.
    if approach == "opt":
        return float(h_train[-1]["loss"]) if h_train else float("nan")
    return float(h_val[-1]["loss"]) if h_val else float("nan")


if __name__ == "__main__":
    main()
