"""Optuna hyperparameter tuning for the opt (KKT) experiments.

Separate from the exhaustive grid (run.py + conf/sweep). ONE process = ONE Optuna
WORKER attached to a shared study (JournalStorage file), so N Slurm array tasks can
tune the SAME study concurrently without a giant job array -- sidestepping both
MaxArraySize and QOSMaxSubmitJobPerUser (see scripts/tune.sh). Reuses run.py's
building blocks (build_data / build_task / build_algorithm / train), always with
approach='opt' (full dataset as train, no val/test -- so the only randomness axis
is init_seed; there are no folds).

Per-trial objective -- the composite KKT residual, for every CONSTRAINED method:
    r = ||grad_x L|| + relu(max_viol) + |compl|
computed per epoch, then tail-averaged over the last --tail epochs and averaged over
--init-seeds (opt mode has no folds). This is EXACTLY plotting/plot_kkt's residual
(_residual_traj), so tuning optimizes precisely what the KKT plots rank configs on.
The compl term is absent (-> 0) for ssg (no duals), leaving r = ||grad_x f|| +
relu(max_viol). (Chosen over minimizing the Lagrangian L directly, which can reward
tiny-lambda configs that sit low on f while still infeasible.)

EXCEPTION: adam (updater='plain') is the UNCONSTRAINED reference -- it is tuned on
plain train loss, ignoring feasibility, so it stays a genuine "ignore the constraints"
baseline rather than being pushed toward feasible regions by the residual.

After the trials, the best config is re-run at full CV (all --init-seeds) and its
per-epoch histories are written in run.py's multirun layout, so aggregate.py /
select_best.py / plot_kkt pick it up as one more config in the (task, data, algo) cell.

Usage
-----
Single local worker (optimize + finalize):
    python tune.py --algo pbm --n-trials 50
One Slurm worker of many (optimize only), then a finalize job (see scripts/tune.sh):
    python tune.py --algo pbm --n-trials 25 --study E1opt_pbm_income \
        --storage selection/opt/tuning/E1opt_pbm_income.log --no-finalize
    python tune.py --algo pbm --study ... --storage ... --n-trials 0   # finalize best
"""
import argparse
import json
import os
from copy import deepcopy

import numpy as np
import optuna
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import algorithms as algo_mod
import data as data_mod
import tasks as tasks_mod
from run import _write_history
from search_spaces import suggest
from train import train

CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
_BIG = 1e6  # objective returned for a diverged / NaN run so the sampler avoids it


def _make_storage(path):
    """Storage shared by the parallel workers. Prefers JournalStorage (optuna>=3.1),
    the robust file-based backend for concurrent access; falls back to a SQLite
    RDBStorage on very old optuna (fine for one local worker, but SQLite over NFS can
    hit lock errors with many concurrent workers -- use a recent optuna on the cluster)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        from optuna.storages import JournalStorage
        try:  # optuna >= 4.0
            from optuna.storages.journal import JournalFileBackend
        except ImportError:  # optuna 3.1 - 3.x
            from optuna.storages import JournalFileStorage as JournalFileBackend
        return JournalStorage(JournalFileBackend(path))
    except ImportError:
        db = path if path.endswith(".db") else path + ".db"
        print(f"[tune] JournalStorage unavailable (old optuna); using sqlite:///{db}")
        return optuna.storages.RDBStorage(f"sqlite:///{db}")


def _base_cfg(algo, data, task, n_epochs):
    """Compose the same config run.py would, pinned to approach='opt'."""
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CONF_DIR, version_base=None)
    return compose(config_name="config",
                   overrides=[f"algorithm={algo}", f"data={data}", f"task={task}",
                              "approach=opt", f"n_epochs={n_epochs}", "verbose=false"])


def _apply(base_cfg, overrides):
    """Deep-copy the base cfg and apply {dotted_key: value} sampled overrides."""
    cfg = deepcopy(base_cfg)
    OmegaConf.set_struct(cfg, False)
    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val)
    return cfg


def _run_one(cfg, init_seed, device):
    """Train one (config, init_seed) in opt mode; mirrors run.py's per-run block.
    Returns (algorithm, task, h_train, h_opt)."""
    torch.manual_seed(init_seed)
    cfg_data = OmegaConf.to_container(cfg.data, resolve=True)
    bundle = data_mod.build_data(
        cfg_data, batch_size=int(cfg.task.batch_size), device=device,
        cv_seed=int(cfg.cv_seed), n_folds=int(cfg.n_folds), fold=int(cfg.fold),
        init_seed=init_seed, approach="opt",
        opt_eval_size=int(cfg.get("opt_eval_size", 10000)))
    task = tasks_mod.build_task(OmegaConf.to_container(cfg.task, resolve=True), bundle)
    torch.manual_seed(init_seed)  # reseed just before init so weights are reproducible
    model = task.model_factory()
    algorithm = algo_mod.build_algorithm(
        cfg.algorithm, model, m=task.m, epoch_length=len(bundle.train_loader))
    h_train, _h_val, _h_test, h_opt = train(
        model, algorithm, task, bundle, n_epochs=int(cfg.n_epochs), device=device,
        approach="opt", verbose=False)
    return algorithm, task, h_train, h_opt


def _train_loss(h_train, tail):
    """Tail-averaged train loss -- the objective for the unconstrained adam reference."""
    vals = [r["loss"] for r in h_train[-tail:] if "loss" in r]
    return float(np.mean(vals)) if vals else float("nan")


def _kkt_residual(h_opt, tail):
    """The scalar this (config, seed) contributes: the composite KKT residual
    r = ||grad_x L|| + relu(max_viol) + |compl|, formed per epoch (like
    plot_kkt._residual_traj) then averaged over the last ``tail`` epochs. The
    max_viol / compl terms drop out when a metric is absent (ssg / adam have no
    compl). NaN if no usable opt rows (diverged / opt eval unavailable)."""
    res = []
    for r in h_opt[-tail:]:
        gn = r.get("grad_norm")
        if gn is None or not np.isfinite(gn):
            continue
        val = float(gn)
        mv, cp = r.get("max_viol"), r.get("compl")
        if mv is not None and np.isfinite(mv):
            val += max(float(mv), 0.0)
        if cp is not None and np.isfinite(cp):
            val += abs(float(cp))
        res.append(val)
    return float(np.mean(res)) if res else float("nan")


def _make_objective(base_cfg, algo, init_seeds, tail, device):
    def objective(trial):
        overrides = suggest(trial, algo)
        # stash the resolved dotted-key overrides so finalize can replay the winner
        # without reconstructing them from trial.params (which lose the mapped values).
        trial.set_user_attr("overrides", overrides)
        cfg = _apply(base_cfg, overrides)
        scores = []
        for seed in init_seeds:
            algorithm, _task, h_train, h_opt = _run_one(cfg, seed, device)
            # adam (plain, unconstrained reference) is tuned on loss; the constrained
            # methods on the KKT residual.
            scores.append(_train_loss(h_train, tail) if algorithm.updater == "plain"
                          else _kkt_residual(h_opt, tail))
        score = float(np.mean(scores))
        return score if np.isfinite(score) else _BIG
    return objective


def _write_run(out_dir, cfg, task, algorithm, h_train, h_opt, algo, data_name, tsk_name,
               init_seed, fold, approach="opt"):
    """Write one run's histories + run_meta.json exactly like run.py, so aggregate.py
    treats the tuned config as another run in the (task, data, algo) cell."""
    os.makedirs(out_dir, exist_ok=True)
    _write_history(h_train, os.path.join(out_dir, "train.csv"))
    _write_history(h_opt, os.path.join(out_dir, "opt.csv"))
    meta = {
        "task": tsk_name, "data": data_name, "algorithm": algo, "approach": approach,
        "init_seed": init_seed, "fold": fold, "n_folds": int(cfg.n_folds),
        "cv_seed": int(cfg.cv_seed), "m": task.m, "bound": task.bound,
        "select_filter": algorithm.select_filter,
        "hyperparameters": OmegaConf.to_container(cfg.algorithm, resolve=True),
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _finalize(study, base_cfg, args, init_seeds, device):
    """Re-run the best config at full CV and write it into the multirun tree."""
    best = study.best_trial
    overrides = best.user_attrs.get("overrides")
    if overrides is None:
        raise RuntimeError("best trial has no stored 'overrides' user_attr; cannot finalize.")
    cfg = _apply(base_cfg, overrides)
    print(f"[finalize] best trial #{best.number}  value={best.value:.6g}\n"
          f"           overrides={overrides}")
    run_root = os.path.join(args.out_runs, args.task, args.algo, "opt")
    for seed in init_seeds:
        algorithm, task, h_train, h_opt = _run_one(cfg, seed, device)
        out_dir = os.path.join(run_root, f"tuned_fold0_init{seed}")
        _write_run(out_dir, cfg, task, algorithm, h_train, h_opt,
                   args.algo, args.data, args.task, init_seed=seed, fold=0)
        print(f"[finalize] wrote {out_dir}")
    with open(os.path.join(run_root, "best_params.json"), "w") as f:
        json.dump({"study": args.study, "value": best.value,
                   "overrides": overrides, "params": best.params}, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", required=True,
                    help="algorithm config group (adam, pbm, pbm_logscaled, alm_max, alm_proj, ssg)")
    ap.add_argument("--data", default="income")
    ap.add_argument("--task", default="folktables_positive_rate_pair")
    ap.add_argument("--n-trials", type=int, default=50, help="trials THIS worker runs")
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--tail", type=int, default=5,
                    help="window (last N epochs) the per-run KKT residual averages over")
    ap.add_argument("--init-seeds", default="0,1,2",
                    help="comma-separated init seeds averaged per trial (opt mode has no folds)")
    ap.add_argument("--study", default=None, help="Optuna study name (default E1opt_<algo>_<data>)")
    ap.add_argument("--storage", default=None,
                    help="JournalStorage file (default selection/opt/tuning/<study>.log)")
    ap.add_argument("--finalize", dest="finalize", action="store_true", default=True,
                    help="after trials, re-run the best config at full CV (default on)")
    ap.add_argument("--no-finalize", dest="finalize", action="store_false",
                    help="skip finalize (use on parallel workers; run a single finalize job after)")
    ap.add_argument("--out-runs", default="multirun",
                    help="multirun root the finalized best config is written under")
    args = ap.parse_args()

    study_name = args.study or f"E1opt_{args.algo}_{args.data}"
    storage_path = args.storage or os.path.join("selection", "opt", "tuning", f"{study_name}.log")
    init_seeds = [int(s) for s in args.init_seeds.split(",") if s.strip() != ""]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    base_cfg = _base_cfg(args.algo, args.data, args.task, args.n_epochs)

    storage = _make_storage(storage_path)
    # constant_liar: with N workers sharing the study, up to N trials are RUNNING (not
    # yet COMPLETE) when a worker samples, so TPE can't condition on them. This flag
    # makes the sampler treat running trials as bad values -> concurrent workers spread
    # out instead of all chasing the same region. No sampler seed: each worker should
    # draw an independent startup sequence (a shared seed would collide across workers).
    sampler = optuna.samplers.TPESampler(constant_liar=True)
    study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler,
                                direction="minimize", load_if_exists=True)

    if args.n_trials > 0:
        print(f"[tune] study={study_name} algo={args.algo} data={args.data} task={args.task}\n"
              f"[tune] {args.n_trials} trials, {len(init_seeds)} seeds/trial {init_seeds}, "
              f"tail={args.tail}, device={device}")
        study.optimize(_make_objective(base_cfg, args.algo, init_seeds, args.tail, device),
                       n_trials=args.n_trials)

    if args.finalize:
        _finalize(study, base_cfg, args, init_seeds, device)


if __name__ == "__main__":
    main()
