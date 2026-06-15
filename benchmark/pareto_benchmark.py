"""
pareto_benchmark.py

Budget-aware Pareto benchmark layer on top of the existing benchmark harness.

Question: "Given a fresh problem and a fixed wall-clock budget T, how good a
(loss, constraint-violation) solution can each method deliver, *including* the
time spent on hyperparameter search?"

Protocol (per method, per wall-clock budget T, per seed):
  1. Random-search HP configs (sampled from that method's space), training each
     to a fixed epoch count via the existing `run_train`, accumulating wall-clock
     from each trial's logged `time` until the budget T is exhausted.
     -> Budget buys BREADTH of search; equal *wall-clock* per method, so a slower
        method (e.g. SPBM, ~3x/epoch) honestly fits fewer trials.
  2. Select the best trial by VALIDATION loss subject to validation feasibility
     (max constraint <= bound * (1 + VAL_FEAS_TOL)).
  3. Record that trial's train AND test (loss, max-violation).
Averaging over seeds gives mean +/- std points that migrate toward the origin
as T grows -> the "Pareto-over-time" plane plot.

This file reuses, without modification:
  - run_train (benchmark_utils)         : single train run, returns histories
  - the constraint construction pattern from run_benchmark.py
  - the data loaders in _data_sources

It adds only: Method specs + random-config samplers (this piece), then the
budget loop, validation selection, and the plane plot (later pieces).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from humancompatible.train.dual_optim import ALM, PBM


# ── protocol constants ───────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]   # 5 outer repeats of the full search-and-select procedure

# Wall-clock budgets (seconds) per (method, seed) HPO run. Placeholders —
# calibrate after timing one trial on the target experiment.
TIME_BUDGETS = [10.0, 30.0, 60.0, 120.0]

EPOCHS_PER_TRIAL = 20          # fixed training length per random-search trial
VAL_FEAS_TOL = 0.10            # validation feasibility slack for selection (paper's +10%)


# ── method specification ─────────────────────────────────────────────────────
@dataclass
class Method:
    """An optimizer family + how to sample a random HP config for it.

    `run_train_kwargs` are the fixed structural args that select the training
    mode in the existing harness (mode / optimizers / slack / eq), exactly as
    run_benchmark.py passes them. `sample_config` returns a param_set dict using
    the prefix convention run_train expects: primal__*, dual__* / dual_*,
    moreau__*. For Adam-with-regularization, the penalty is passed separately as
    `reg_penalty`, so sample_config may also emit a special '__reg_penalty' key
    that the loop pulls out (handled later).
    """
    name: str                              # internal id, e.g. "pbm"
    display: str                           # plot label, e.g. "SPBM"
    color: str
    marker: str
    # structural kwargs forwarded to run_train (mode-selecting, not searched):
    run_train_kwargs: Dict
    # rng -> param_set dict (the searched hyperparameters):
    sample_config: Callable[[np.random.Generator], Dict]


# ── random-config samplers (grids from SPBM paper Appendix D.1) ──────────────
# Common objective learning rate grid for E1-E4 (Adam/Lagrangian primal step).
_PRIMAL_LR_GRID = [0.0001, 0.0005, 0.001, 0.005]


# All methods except E1 carry slight L2 weight decay (Appendix D.1).
_WEIGHT_DECAY = 0.01


def _sample_adam(rng) -> Dict:
    """Unconstrained Adam. Real config keys: primal__lr, primal__weight_decay.
    Paper grid: lr in {1e-4, 5e-4, 1e-3, 5e-3}."""
    return {
        "primal__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "primal__weight_decay": _WEIGHT_DECAY,
    }


def _sample_alm(rng) -> Dict:
    """SSL-ALM. Real config keys (alm_max/alm_slack): primal__lr, dual__lr,
    dual__penalty, moreau__mu, primal__weight_decay. Paper grid: primal & dual
    lr in the 4-value grid; penalty rho in {0, 1, 2}; Moreau mu in {0.1, 1, 2}."""
    return {
        "primal__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "dual__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "dual__penalty": float(rng.choice([0.0, 1.0, 2.0])),
        "moreau__mu": float(rng.choice([0.1, 1.0, 2.0])),
        "primal__weight_decay": _WEIGHT_DECAY,
    }


def _sample_pbm(rng) -> Dict:
    """SPBM. Real config keys (pbm_dimin): primal__lr, dual__gamma,
    dual__penalty_update, dual__pbf, dual__penalty_mult, dual__delta,
    init_duals, moreau__mu, primal__weight_decay. Paper grid (E1-E4): primal lr
    in 4-value grid; gamma in {0.1,0.5,0.9,0.95,0.99,0.999,0.9999}; diminishing
    penalty update with K1 in {0.99,0.999,0.9999,1}; Quadratic-Logarithmic
    penalty/barrier; Moreau mu in {0.1,1,2}."""
    return {
        "primal__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "dual__gamma": float(rng.choice([0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999])),
        "dual__penalty_update": "dimin",
        "dual__pbf": "quadratic_logarithmic",
        "dual__penalty_mult": float(rng.choice([0.99, 0.999, 0.9999, 1.0])),
        "dual__delta": 1.0,
        "moreau__mu": float(rng.choice([0.1, 1.0, 2.0])),
        "primal__weight_decay": _WEIGHT_DECAY,
    }


def _sample_ssg(rng) -> Dict:
    """SSw (switching subgradient). Follows the primal/dual lr pattern; objective
    and constraint (dual) lr both in the 4-value grid; Moreau mu in {0.1,1,2}.
    NOTE: dual__* keys here feed a plain torch.optim.Adam as the 'dual' optimizer
    (mode='sw'); confirm SSG's expected keys against your conf/algorithm/ssg yaml."""
    return {
        "primal__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "dual__lr": float(rng.choice(_PRIMAL_LR_GRID)),
        "moreau__mu": float(rng.choice([0.1, 1.0, 2.0])),
        "primal__weight_decay": _WEIGHT_DECAY,
    }


# ── method registry ──────────────────────────────────────────────────────────
# Structural kwargs mirror exactly how run_benchmark.py invokes run_train for
# each algorithm (mode, primal/dual optimizer, slack, eq, fuse_loss_constraint).
METHODS = {
    "adam": Method(
        name="adam", display="Unconstrained Adam", color="tab:blue", marker="o",
        run_train_kwargs=dict(
            primal_opt=torch.optim.Adam, dual_opt=None, mode="torch",
            constraints_to_eq=False, use_slack=False,
        ),
        sample_config=_sample_adam,
    ),
    "pbm": Method(
        name="pbm", display="SPBM", color="tab:purple", marker="v",
        run_train_kwargs=dict(
            primal_opt=torch.optim.Adam, dual_opt=PBM, mode="hc",
            constraints_to_eq=False, use_slack=False,
        ),
        sample_config=_sample_pbm,
    ),
    "alm": Method(
        name="alm", display="SSL-ALM", color="tab:orange", marker="s",
        run_train_kwargs=dict(
            primal_opt=torch.optim.Adam, dual_opt=ALM, mode="hc",
            constraints_to_eq=True, use_slack=True,
        ),
        sample_config=_sample_alm,
    ),
    "ssg": Method(
        name="ssg", display="SSw", color="tab:green", marker="^",
        run_train_kwargs=dict(
            primal_opt=torch.optim.Adam, dual_opt=torch.optim.Adam, mode="sw",
            constraints_to_eq=False, use_slack=False,
        ),
        sample_config=_sample_ssg,
    ),
}


# ── problem bundle (the seam you wire from your config) ──────────────────────
@dataclass
class Problem:
    """Everything run_train needs that is *not* method- or HP-specific.

    You construct this once per experiment from your Hydra config (data loader +
    constraint as in run_benchmark.py). The benchmark engine treats it as opaque
    and forwards the fields straight into run_train.

    Fields mirror run_train's non-HP arguments exactly:
      m                  : number of scalar constraints (= c.m_fn(n_groups))
      data_train         : (features_train, sens_train, labels_train) tensors
      dataloader         : training DataLoader
      data_val           : (feat, sens, lab) tuple OR a val DataLoader
      data_test          : (feat, sens, lab) tuple OR a test DataLoader  [for reporting]
      constraint_fn      : c.compute_constraints
      constraint_bound   : epsilon threshold (also the plot feasibility line)
      fuse_loss_constraint : True for Loss* constraints (CIFAR equal-loss), else False
    """
    m: int
    data_train: tuple
    dataloader: object
    data_val: object
    data_test: object
    constraint_fn: Callable
    constraint_bound: float
    fuse_loss_constraint: bool = False
    name: str = "experiment"


# ── extracting a single trial's outcome from run_train histories ─────────────
def _last_epoch_point(history: List[dict], m: int):
    """Collapse a run_train history (list of per-log dicts) to the final
    (loss, max_violation, wall_time) for that split.

    history rows carry: 'loss', 'c_0'..'c_{m-1}', 'time'. We take the last
    logged row (end of training) and the max over the m constraint columns.
    """
    if not history:
        return float("inf"), float("inf"), 0.0
    row = history[-1]
    loss = float(row["loss"])
    viols = [float(row[f"c_{j}"]) for j in range(m) if f"c_{j}" in row]
    max_viol = max(viols) if viols else float("inf")
    wall = float(row.get("time", 0.0))
    return loss, max_viol, wall


@dataclass
class TrainResult:
    """Raw metrics from ONE (config, seed) training run."""
    tr_loss: float
    tr_viol: float
    val_loss: float
    val_viol: float
    te_loss: float
    te_viol: float
    wall_time: float


@dataclass
class ConfigEval:
    """STRICT evaluation of one config across ALL seeds.

    selection_* are the across-seed MEAN validation metrics used to choose the
    best config (so we never crown a config that merely got lucky on one seed).
    The per-seed train/test arrays are kept so the reported point's error bars
    are the spread of THIS config (same config every seed) — which is what makes
    the std meaningful.
    """
    config: dict
    sel_val_loss: float          # mean val loss over seeds (selection key)
    sel_val_viol: float          # mean val max-viol over seeds (feasibility key)
    tr_loss: np.ndarray          # per-seed train loss
    tr_viol: np.ndarray
    te_loss: np.ndarray          # per-seed test loss
    te_viol: np.ndarray
    wall_time: float             # TOTAL wall-clock for this config = sum over seeds


# ── one (config, seed) training run ──────────────────────────────────────────
def _train_once(method: Method, config: dict, problem: Problem, seed: int) -> TrainResult:
    """Train a single config under a single seed via the existing run_train, then
    read train/val from its histories and evaluate test on the held-out set.

    seed drives torch's init + batch order for THIS run. Wall-clock is taken from
    run_train's own tracker (which excludes paused constraint-eval for the
    unconstrained case — exactly the time we want to bill)."""
    from benchmark_utils import run_train, eval as _eval, eval_loader

    torch.manual_seed(seed)

    model, train_hist, val_hist = run_train(
        m=problem.m,
        param_set=config,
        data_train=problem.data_train,
        dataloader=problem.dataloader,
        data_val=problem.data_val,
        n_epochs=EPOCHS_PER_TRIAL,
        c_fn=problem.constraint_fn,
        constraint_bound=problem.constraint_bound,
        verbose=False,
        fuse_loss_constraint=problem.fuse_loss_constraint,
        **method.run_train_kwargs,
    )

    tr_loss, tr_viol, tr_wall = _last_epoch_point(train_hist, problem.m)
    val_loss, val_viol, _ = _last_epoch_point(val_hist, problem.m)

    if isinstance(problem.data_test, tuple):
        te_loss, te_viols = _eval(model, problem.data_test, problem.constraint_fn)
    else:
        te_loss, te_viols = eval_loader(model, problem.data_test, problem.constraint_fn)
    te_viol = max(te_viols) if te_viols else float("inf")

    return TrainResult(
        tr_loss=tr_loss, tr_viol=tr_viol,
        val_loss=val_loss, val_viol=val_viol,
        te_loss=float(te_loss), te_viol=float(te_viol),
        wall_time=tr_wall,
    )


# ── STRICT: evaluate one config across all seeds ─────────────────────────────
def evaluate_config(method: Method, config: dict, problem: Problem,
                    seeds=SEEDS) -> ConfigEval:
    """Train `config` on EVERY seed; selection metrics are the across-seed means.

    This is the strict protocol: a config's quality is its MEAN validation
    performance over seeds, so selection cannot be fooled by single-seed luck.
    The full wall-clock cost (sum over seeds) is what the budget will be billed.
    """
    runs = [_train_once(method, config, problem, s) for s in seeds]
    tr_loss = np.array([r.tr_loss for r in runs])
    tr_viol = np.array([r.tr_viol for r in runs])
    val_loss = np.array([r.val_loss for r in runs])
    val_viol = np.array([r.val_viol for r in runs])
    te_loss = np.array([r.te_loss for r in runs])
    te_viol = np.array([r.te_viol for r in runs])
    total_wall = float(sum(r.wall_time for r in runs))

    return ConfigEval(
        config=config,
        sel_val_loss=float(val_loss.mean()),
        sel_val_viol=float(val_viol.mean()),
        tr_loss=tr_loss, tr_viol=tr_viol,
        te_loss=te_loss, te_viol=te_viol,
        wall_time=total_wall,
    )


# ── one budget-bounded random search (STRICT) ────────────────────────────────
def run_budget_search(method: Method, problem: Problem, budget_s: float,
                      master_seed: int, seeds=SEEDS):
    """Sample configs and evaluate each on ALL seeds, accumulating the full
    (sum-over-seeds) wall-clock, until the budget is exhausted. Then select the
    config with the best MEAN validation quality.

    master_seed seeds the config-sampling stream only (which configs get drawn);
    the per-config training always uses the fixed `seeds` set, so selection is
    over mean-across-seeds performance and the budget is billed the true cost of
    that robust evaluation.

    Selection: among configs whose MEAN val max-viol <= bound*(1+tol), pick the
    lowest MEAN val loss; if none feasible, pick the smallest mean val viol.

    Returns (selected_eval, all_evals, total_wall).
    """
    rng = np.random.default_rng(master_seed)
    evals: List[ConfigEval] = []
    spent = 0.0

    while spent < budget_s or not evals:
        config = method.sample_config(rng)
        ce = evaluate_config(method, config, problem, seeds=seeds)
        evals.append(ce)
        spent += ce.wall_time          # full sum-over-seeds cost billed here

    feas_thresh = problem.constraint_bound * (1.0 + VAL_FEAS_TOL)
    feasible = [e for e in evals if e.sel_val_viol <= feas_thresh]
    if feasible:
        selected = min(feasible, key=lambda e: e.sel_val_loss)
    else:
        selected = min(evals, key=lambda e: e.sel_val_viol)

    return selected, evals, spent


# ── aggregation: one (method, budget) -> one mean+/-std point ─────────────────
@dataclass
class BudgetPoint:
    """Aggregated result for a single (method, budget).

    The mean/std are over the SAME selected config's per-seed runs (strict
    protocol) — so the error bars are the spread of the chosen configuration,
    not of 'which config won'. Diagnostics record how many distinct configs the
    budget allowed and whether the chosen config was feasible on mean validation.
    """
    budget_s: float
    n_seeds: int
    tr_loss_mean: float; tr_loss_std: float
    tr_viol_mean: float; tr_viol_std: float
    te_loss_mean: float; te_loss_std: float
    te_viol_mean: float; te_viol_std: float
    n_configs: float            # distinct configs evaluated within the budget
    feasible: float             # 1.0 if chosen config feasible on mean val, else 0.0


def aggregate_budget(method: Method, problem: Problem, budget_s: float,
                     master_seed: int = 0, seeds=SEEDS, verbose=True) -> BudgetPoint:
    """Run ONE strict budget search; the reported point is the selected config's
    per-seed mean +/- std (train and test).

    Note: unlike the old design there is no outer seed loop here — the seed
    averaging happens INSIDE selection (every config is scored on all seeds).
    master_seed only varies which configs are sampled; for a single headline
    point use a fixed master_seed. (If you later want variance over the whole
    search procedure too, wrap this over several master_seeds — but that
    multiplies cost and is usually unnecessary given the strict selection.)
    """
    selected, evals, spent = run_budget_search(
        method, problem, budget_s, master_seed, seeds=seeds
    )
    feas_thresh = problem.constraint_bound * (1.0 + VAL_FEAS_TOL)
    feasible = float(selected.sel_val_viol <= feas_thresh)

    if verbose:
        print(f"    [{method.name}] T={budget_s:>6.0f}s  configs={len(evals):>3}  "
              f"chosen feasible={bool(feasible)}  "
              f"te_loss={selected.te_loss.mean():.4f}±{selected.te_loss.std():.4f}  "
              f"te_viol={selected.te_viol.mean():.4f}±{selected.te_viol.std():.4f}")

    return BudgetPoint(
        budget_s=budget_s, n_seeds=len(seeds),
        tr_loss_mean=float(selected.tr_loss.mean()), tr_loss_std=float(selected.tr_loss.std()),
        tr_viol_mean=float(selected.tr_viol.mean()), tr_viol_std=float(selected.tr_viol.std()),
        te_loss_mean=float(selected.te_loss.mean()), te_loss_std=float(selected.te_loss.std()),
        te_viol_mean=float(selected.te_viol.mean()), te_viol_std=float(selected.te_viol.std()),
        n_configs=float(len(evals)),
        feasible=feasible,
    )


def sweep_method(method: Method, problem: Problem, budgets=TIME_BUDGETS,
                 seeds=SEEDS, verbose=True) -> List[BudgetPoint]:
    """All budgets for one method -> a trajectory of BudgetPoints (the line that
    migrates toward the origin as budget grows)."""
    points = []
    for T in budgets:
        if verbose:
            print(f"  {method.display}: budget {T:.0f}s")
        points.append(aggregate_budget(method, problem, T, seeds=seeds, verbose=verbose))
    return points


# ── persistence ──────────────────────────────────────────────────────────────
import csv
import os

_POINT_FIELDS = [
    "budget_s", "n_seeds",
    "tr_loss_mean", "tr_loss_std", "tr_viol_mean", "tr_viol_std",
    "te_loss_mean", "te_loss_std", "te_viol_mean", "te_viol_std",
    "n_configs", "feasible",
]


def save_points(points: List[BudgetPoint], method_name: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"budget_pareto_{method_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_POINT_FIELDS)
        w.writeheader()
        for p in points:
            w.writerow({k: getattr(p, k) for k in _POINT_FIELDS})
    print(f"  saved {path}")


def load_points(method_name: str, results_dir: str) -> Optional[List[BudgetPoint]]:
    path = os.path.join(results_dir, f"budget_pareto_{method_name}.csv")
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows:
            vals = {k: (int(float(r[k])) if k == "n_seeds" else float(r[k]))
                    for k in _POINT_FIELDS}
            out.append(BudgetPoint(**vals))
        print(f"  loaded cached {path}")
        return out
    except FileNotFoundError:
        return None


# ── the migrating-points plane plot ──────────────────────────────────────────
def _draw_plane(ax, all_series, loss_mean_k, loss_std_k, viol_mean_k, viol_std_k,
                bound, title, annotate_budget=True):
    import matplotlib.pyplot as plt

    for points, method in all_series:
        viol   = [getattr(p, viol_mean_k) for p in points]
        loss   = [getattr(p, loss_mean_k) for p in points]
        viol_e = [getattr(p, viol_std_k)  for p in points]
        loss_e = [getattr(p, loss_std_k)  for p in points]

        ax.errorbar(
            viol, loss, xerr=viol_e, yerr=loss_e,
            fmt=method.marker + "-", color=method.color, label=method.display,
            capsize=3, markersize=6, linewidth=1.3, alpha=0.85,
        )
        if annotate_budget:
            for p, x, y in zip(points, viol, loss):
                ax.annotate(f"{p.budget_s:.0f}s", (x, y), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color=method.color)

    # feasibility threshold (constraint <= bound): vertical line
    ax.axvline(bound, ls="--", color="red", lw=1, alpha=0.7, label="threshold")
    ax.set_xlabel("max constraint violation  (lower is fairer)")
    ax.set_ylabel("BCE loss  (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_budget_pareto(all_series, bound, out="budget_pareto.svg", suptitle=None):
    """all_series: list of (List[BudgetPoint], Method). Two panels: train | test.
    Each method is one trajectory; points move toward the origin as budget grows.
    """
    import matplotlib.pyplot as plt

    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _draw_plane(ax_tr, all_series, "tr_loss_mean", "tr_loss_std",
                "tr_viol_mean", "tr_viol_std", bound, "Train")
    _draw_plane(ax_te, all_series, "te_loss_mean", "te_loss_std",
                "te_viol_mean", "te_viol_std", bound, "Test")
    fig.suptitle(suptitle or
                 "Budget-aware Pareto: quality vs. constraint violation as a\n"
                 "function of total practitioner time (search + train)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved plot to {out}")
    return fig


# ── orchestration ────────────────────────────────────────────────────────────
def run_budget_benchmark(problem: Problem, method_names=("adam", "pbm", "alm", "ssg"),
                         budgets=TIME_BUDGETS, seeds=SEEDS,
                         results_dir="budget_results", use_cache=True):
    """Top-level: for each method, sweep budgets (with caching), then plot.
    `problem` is built by you from your config (see build_problem stub below)."""
    results_dir = os.path.join(results_dir, problem.name)
    all_series = []
    for name in method_names:
        method = METHODS[name]
        print(f"\n=== {method.display} ===")
        points = load_points(name, results_dir) if use_cache else None
        if points is None:
            points = sweep_method(method, problem, budgets=budgets, seeds=seeds)
            save_points(points, name, results_dir)
        all_series.append((points, method))

    out = os.path.join(results_dir, "budget_pareto.svg")
    plot_budget_pareto(all_series, problem.constraint_bound, out=out,
                       suptitle=f"Budget-aware Pareto ({problem.name})")
    return all_series


def build_problem(experiment: str = "E2") -> Problem:
    """STUB — you wire this from your Hydra config / data loaders.

    Build the data + constraint exactly as run_benchmark.py does, then pack into
    a Problem. Reference (paraphrased from run_benchmark.py):

        # 1. load data via _data_sources (e.g. load_data_FT) -> dataloaders + tensors
        # 2. construct the constraint object c (e.g. FairretPairwise / FairretAgg)
        #    from your constraint_cfg, compute m = c.m_fn(sens_train.shape[-1])
        # 3. set constraint_bound = constraint_cfg['bound'], and
        #    fuse_loss_constraint = constraint name starts with 'Loss'
        # 4. choose data_test (held-out) for reporting — run_benchmark.py reuses
        #    test as val for some tasks; here we want a TRUE test split distinct
        #    from the val set used for config selection.

    Return:
        Problem(m=..., data_train=(feat,sens,lab), dataloader=...,
                data_val=(feat,sens,lab) or loader,
                data_test=(feat,sens,lab) or loader,
                constraint_fn=c.compute_constraints,
                constraint_bound=..., fuse_loss_constraint=...,
                name=experiment)
    """
    raise NotImplementedError(
        "Wire build_problem() to your data loaders + constraint construction "
        "(mirror run_benchmark.py). The engine above is ready to consume it."
    )


if __name__ == "__main__":
    problem = build_problem("E2")
    run_budget_benchmark(problem)