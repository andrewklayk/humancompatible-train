"""
Hyperparameter sweeps for E2 and E3.

One invocation = **one** (experiment, algorithm, hyperparameter) combination, writing one
``row.json`` into its own Hydra job directory. ``-m`` produces the cartesian product and
``hydra/launcher=slurm`` spreads it over the cluster. Nothing here runs a grid itself.

This exists because every knob worth tuning in ``paper/`` is currently unreachable from
outside its module: E2a's dual step is the module constant ``DUAL_LR`` closed over by the
``METHODS`` builders, and E3's is a single scalar flag. E2a's docstring says the tuning
burden "needs its own sweep; it is not answered here" -- this is that sweep.

**What it does not do.** It does not replace the existing drivers. ``a_fairness.py``,
``e3/sweep.py``, ``scaling.py`` and ``run_cifar.py`` keep their CLIs and keep producing the
committed artifacts under ``paper/results/``; they own the *configuration* grids (method x
eps x seed), this owns the *hyperparameter* grids. It never writes to ``paper/results/``:
every job redirects the shared harness at its own job directory (see :func:`_setup`).

Usage::

    python paper/tune.py experiment=e2a algorithm=alm
    python paper/tune.py -m experiment=e2a algorithm=alm +sweep=e2a_alm
    python paper/tune.py -m hydra/launcher=slurm experiment=e2a +sweep=e2a_alm seed=0,1,2

Then rank the results with ``python paper/tune_report.py --runs paper/multirun/e2a/alm``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from paper import _harness

# A configuration far enough from feasible that no achievable objective can outrank it.
# Sweeps are ranked on a single scalar, so infeasibility has to enter it as a penalty
# rather than as a filter, or the best-looking row is the one that ignored the constraint.
INFEASIBLE_WEIGHT = 1e3


# --------------------------------------------------------------------------- #
# config plumbing
# --------------------------------------------------------------------------- #


def _cfg_hash(*parts) -> str:
    """Short, stable hash of the config groups that identify one sweep cell.

    Used as ``hydra.sweep.subdir``. ``${hydra.job.num}`` would not do: it restarts on
    every ``-m``, so two different grids would write into the same directories and a
    re-run would not be idempotent. Mirrors ``new_bench/run.py:_hp_hash``.
    """
    blob = json.dumps(
        [OmegaConf.to_container(part, resolve=True) for part in parts],
        sort_keys=True, default=str,
    )
    return hashlib.md5(blob.encode()).hexdigest()[:10]


# Registered at import, not inside main(): Hydra resolves the sweep subdir on the
# launching side *and* again inside each submitit task, which imports this module.
OmegaConf.register_new_resolver("cfg_hash", _cfg_hash, replace=True)


def _dual_factory(cfg):
    """``m -> DualOptimizer`` from ``cfg.algorithm.dual``, or ``None`` for a reference.

    ``_partial_: true`` in the yaml means :func:`instantiate` returns the constructor with
    its hyperparameters already bound; only ``m`` (and, on GPU, ``device``) is left.
    """
    if cfg.algorithm.get("dual") is None:
        return None
    partial = instantiate(cfg.algorithm.dual)
    return lambda m, **kwargs: partial(m=m, **kwargs)


def _setup(cfg) -> Path:
    """Redirect every artifact this job writes into its own Hydra output directory."""
    out = Path(HydraConfig.get().runtime.output_dir)
    _harness.set_results_dir(out)
    return out


# --------------------------------------------------------------------------- #
# adapter: E2a -- fairness-constrained learning
# --------------------------------------------------------------------------- #


def _run_e2a(cfg) -> dict:
    from paper.e2 import a_fairness
    from paper.problems import fairness

    seed = int(cfg.seed)
    spec = OmegaConf.to_container(cfg.experiment.problem, resolve=True)
    # `states` and `sens_attrs` reach folktables somewhere that hashes them, so a yaml
    # list is not interchangeable with build()'s tuple defaults.
    spec = {key: tuple(value) if isinstance(value, list) else value
            for key, value in spec.items()}
    problem = fairness.build(**spec)
    factory = _dual_factory(cfg)
    label = str(cfg.algorithm.label)

    print(f"{label} on {problem.name}: m={problem.m}, {problem.n_groups} groups, "
          f"batch {problem.batch_size}, {len(problem.loader)} batches/epoch")

    history = a_fairness.run(
        problem,
        # With a factory the registry is bypassed entirely and `method` only names the
        # row; without one this must be a real key, and "Adam" is the reference.
        label if factory is not None else "Adam",
        seed,
        int(cfg.experiment.epochs),
        # a_fairness calls the factory with `m` alone; the problem is on CPU.
        dual_factory=factory,
        primal_lr=float(cfg.algorithm.primal.lr),
    )

    # Tail mean, not the last epoch: a stochastic primal-dual method oscillates around the
    # constraint boundary, so ranking on a single final value ranks configurations by which
    # side of it the last step happened to land on. a_fairness.py settled this.
    tail = history[-min(int(cfg.experiment.tail), len(history) - 1):]

    def mean(key, source=tail):
        return float(np.mean([h[key] for h in source]))

    train_viol, test_viol = mean("train_max_viol"), mean("test_max_viol")
    row = {
        "problem": problem.name,
        "m": problem.m,
        "method": label,
        "seed": seed,
        "tail epochs": len(tail),
        "train loss": mean("train_loss"),
        "test loss": mean("test_loss"),
        "train max_viol": train_viol,
        "test max_viol": test_viol,
        "generalization gap": test_viol - train_viol,
        "train max_viol osc": float(np.std([h["train_max_viol"] for h in tail])),
        "last-epoch train max_viol": history[-1]["train_max_viol"],
        "grad_norm": mean("train_grad_norm"),
        "s/epoch": mean("epoch_time", history[1:]),
    }

    _harness.write_csv(history, "e2a_trajectory", "e2a")
    # Feasibility first: an infeasible configuration is not a cheaper solution, it is a
    # different problem. Judged on train, which is what the method actually optimized.
    row["objective"] = row["test loss"] + INFEASIBLE_WEIGHT * max(0.0, train_viol)
    print(f"  loss {row['test loss']:.4f}  viol train {train_viol:+.4f} "
          f"test {test_viol:+.4f}  -> objective {row['objective']:.4f}")
    return row


# --------------------------------------------------------------------------- #
# adapter: E3 -- CIFAR
# --------------------------------------------------------------------------- #

_E3_CIFAR_KEYS = (
    "dataset", "data", "download", "synthetic", "granularity", "depth", "epochs",
    "batch_size", "steps_per_epoch", "eval_batches", "workers", "quick",
    "lr", "weight_decay", "gate_lr", "init_open",
)


def _run_e3_cifar(cfg) -> dict:
    from paper.e3 import run_cifar

    seed = int(cfg.seed)
    factory = _dual_factory(cfg)
    label = str(cfg.algorithm.label)
    eps = float(cfg.experiment.eps)

    # run_cifar's own defaults, then this config on top -- the idiom e3/sweep.py:72-82
    # already uses, so a default added to build_parser() still reaches the sweep.
    args = run_cifar.build_parser().parse_args([])
    for key in _E3_CIFAR_KEYS:
        setattr(args, key, cfg.experiment[key])
    args.seed = seed
    args.eps = [eps]
    args.penalty = [float(cfg.experiment.penalty)]
    # `none` installs the gates and applies no sparsity pressure; the factory then supplies
    # the pressure. Needed because run_cifar.METHODS drops every builder-less entry of
    # run_llm.METHODS, so "adam" is not a key there -- and because a method commented out
    # of that registry (PBM, nuPI) is still perfectly runnable through a factory.
    name = cfg.algorithm.get("method")
    args.method = [name if name in run_cifar.METHODS else "none"]
    run_cifar.finalize(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_classes = run_cifar.build_loaders(args)

    # run_cifar's registry calls its builders as (m, device, process_group, lr); `lr` is
    # dropped because the step size is already bound into the config's partial, wherever
    # the class happens to call it (lr / beta / ki+kp / penalty_mult).
    if factory is None:
        dual_factory = None
    else:
        def dual_factory(m, dev, process_group, _lr):
            return factory(m, device=dev, process_group=process_group)

    epochs, summary, report = run_cifar.train(
        args, label=label, method=args.method[0], eps=eps,
        penalty=args.penalty[0], train_loader=train_loader, test_loader=test_loader,
        n_classes=n_classes, device=device, dual_factory=dual_factory,
    )

    _harness.write_csv(epochs, "e3_cifar_epochs", "e3")
    if report:
        _harness.write_csv([dict(run=label, **r) for r in report], "e3_cifar_density", "e3")

    row = {"method": label, "eps": eps, "seed": seed, **summary}
    row["objective"] = _density_objective(
        1.0 - float(summary.get("test_acc", 0.0)),
        summary.get("final_density"), eps, factory is not None,
    )
    return row


# --------------------------------------------------------------------------- #
# adapter: E3 -- LM
# --------------------------------------------------------------------------- #

# (config key, flag). Deliberately argv rather than a config object: run_llm.main() may
# re-exec itself under torch.distributed.run, and a DictConfig cannot cross that boundary.
# So the sweepable surface here is exactly build_parser()'s flag set, and
# `algorithm.dual` is ignored -- only `algorithm.method` is read.
_E3_LLM_FLAGS = (
    ("model", "--model"), ("granularity", "--granularity"), ("eps", "--eps"),
    ("penalty", "--penalty"), ("steps", "--steps"), ("warmup", "--warmup"),
    ("eval_batches", "--eval-batches"), ("seq_len", "--seq-len"),
    ("batch_size", "--batch-size"), ("lr", "--lr"), ("gate_lr", "--gate-lr"),
    ("dual_lr", "--dual-lr"), ("grad_clip", "--grad-clip"), ("dtype", "--dtype"),
    ("init_open", "--init-open"), ("ranks", "--ranks"),
)


def _run_e3_llm(cfg) -> dict:
    from paper.e3 import run_llm

    method = cfg.algorithm.get("method")
    if method is None:
        raise SystemExit(
            f"algorithm={cfg.algorithm.name} has no run_llm.METHODS counterpart "
            f"(method: null). The E3-LM adapter drives run_llm through argv, so it can "
            f"only select a method that registry defines; choose from "
            f"{sorted(run_llm.METHODS)} or add the entry."
        )

    seed = int(cfg.seed)
    argv = ["--method", str(method), "--seed", str(seed)]
    for key, flag in _E3_LLM_FLAGS:
        argv += [flag, str(cfg.experiment[key])]
    if cfg.experiment.get("tokens"):
        argv += ["--tokens", str(cfg.experiment.tokens)]
    if cfg.experiment.get("vocab_size") is not None:
        argv += ["--vocab-size", str(cfg.experiment.vocab_size)]
    for key, flag in (("synthetic", "--synthetic"), ("quick", "--quick")):
        if cfg.experiment.get(key):
            argv.append(flag)

    run_llm.main(argv)

    # Read the summary back off disk rather than out of main(), which returns nothing --
    # and cannot, on the multi-rank path where the result lives in a child process. This is
    # the read-back scaling.py already does. `_slug` omits seed/lr/dual_lr, so two tuning
    # runs share a filename; the per-job results dir is what keeps them apart.
    args = run_llm.finalize(run_llm.build_parser().parse_args(argv))
    slug = run_llm._slug(args, int(cfg.experiment.ranks))
    path = _harness.RESULTS / run_llm.EXPERIMENT / f"e3_summary_{slug}.json"
    summary = json.loads(path.read_text())[0]

    row = {"method": str(method), "eps": float(cfg.experiment.eps), "seed": seed, **summary}
    row["objective"] = _density_objective(
        summary.get("eval_ppl"), summary.get("final_density_mean"),
        float(cfg.experiment.eps), method in run_llm.CONSTRAINED,
    )
    return row


def _density_objective(quality, achieved, eps, constrained: bool) -> float:
    """Quality, penalized by the distance from the requested density.

    The **final** iterate, not a tail mean: E3's constraint is a closed form in the gate
    parameters, with no data and no sampling noise in it, so the last value *is* the
    achieved density. Unconstrained references have no target, so they are scored on
    quality alone -- they are a reference, not a competitor.
    """
    if quality is None or not np.isfinite(float(quality)):
        return float("inf")
    quality = float(quality)
    if not constrained or achieved is None or not np.isfinite(float(achieved)):
        return quality
    return quality + INFEASIBLE_WEIGHT * abs(float(achieved) - eps)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

ADAPTERS = {"e2a": _run_e2a, "e3_cifar": _run_e3_cifar, "e3_llm": _run_e3_llm}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    out = _setup(cfg)
    target = str(cfg.experiment.target)
    if target not in ADAPTERS:
        raise SystemExit(f"unknown experiment.target {target!r} "
                         f"(choose from {sorted(ADAPTERS)})")

    row = ADAPTERS[target](cfg)

    # The sidecar post-processing reads, so tune_report.py never has to re-parse a Hydra
    # config -- the same separation new_bench keeps between run.py and aggregate.py.
    record = {
        "experiment": target,
        "algorithm": str(cfg.algorithm.name),
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        **row,
    }
    (out / "row.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"  row    -> {out / 'row.json'}")

    # Returned so a sweeper that expects a float objective stays happy. The basic sweeper
    # ignores it; it is the hook an Optuna sweeper would consume without touching this
    # file. Same choice as new_bench/run.py.
    return float(row["objective"])


if __name__ == "__main__":
    main()
