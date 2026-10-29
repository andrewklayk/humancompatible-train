"""
E2d — stochastic Ghost-penalty SQP on the Dutch census.

The SQP counterpart of the dual methods, on the problem ``benchmark/new_bench``
calls ``dutch_positive_rate_pair`` in its ``approach=opt`` mode, mirrored so the
output can be read next to new_bench's own Dutch runs:

* the **whole dataset is the training set** (StandardScaler fit on all of it), no
  split -- this measures optimization, not generalization;
* 18 groups (sex x age), every ordered pair's positive-rate gap bounded by 0.1,
  so m = 306;
* 4 samples per group per draw (batch 72), BCE reweighted by the inverse group
  propensity (new_bench's ``reweight_loss: true``);
* the 64-32-1 MLP, 40 epochs, float32;
* per epoch, the full-batch metrics of new_bench's ``opt.csv`` at the frozen
  iterate: ``loss`` (unweighted f), ``max_viol``, ``c_j``, and ``grad_norm`` =
  ``||grad f||`` as new_bench logs for its methods without duals.

An "epoch" is new_bench's epoch length, ``len(BalancedBatchSampler)``, so both use
the same number of samples per epoch. The one deviation: Ghost's MLMC estimator is
only unbiased on fresh i.i.d. draws (see ``sqp/ghost/sampler.py``), so batches are
drawn with replacement within each group rather than by permutation.

At m = 306 the constraints are passed as **one vector-valued callable**, so the
Jacobian is one batched backward rather than 306.

Predictions, registered before running:

P1 Ghost ends **feasible** on the full data: tail-mean max-violation <= 0. The
   constant predictor is feasible, so the feasible set is nonempty and a failure
   is the method's, not the problem's.
P2 It does so **without collapsing to a constant predictor**: tail-mean f is
   below its epoch-0 value.

Usage::

    python paper/e2/d_ghost.py --quick
    python paper/e2/d_ghost.py --check
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from humancompatible.train.fairness.utils import BalancedBatchSampler
from humancompatible.train.sqp.ghost import GhostConfig, GhostSQP, Sampler
from paper._harness import (
    ALGORITHM_CONF, Checks, figure, main_exit, save_figure, set_seed, write_csv,
    write_table,
)
from paper.problems import fairness

EXPERIMENT = "e2d"

EPOCHS = 40
TAIL = 5
BOUND = 0.1
PER_GROUP = 4
SEEDS = (0, 1, 2)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_data():
    """Full Dutch data, scaled on all of it (new_bench's opt path), as tensors."""
    features, groups, labels = fairness._load_dutch()
    features = StandardScaler().fit_transform(features)
    return tuple(torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
                 for a in (features, groups, labels))


class BalancedSampler(Sampler):
    """``J / G`` i.i.d. draws per group, laid out in per-group blocks.

    Every group must be present or its positive rate is 0/0. The block layout keeps
    ``split_even_odd``'s halves balanced too, as long as ``J / G`` is even.
    """

    def __init__(self, X, A, y):
        self.X, self.A, self.y = X, A, y
        self.groups = [torch.nonzero(A[:, g]).reshape(-1).numpy()
                       for g in range(A.shape[1])]

    def draw(self, J, rng):
        k, rest = divmod(J, len(self.groups))
        assert rest == 0, f"batch {J} not divisible by {len(self.groups)} groups"
        idx = torch.as_tensor(np.concatenate([rng.choice(g, size=k) for g in self.groups]))
        return self.X[idx], self.A[idx], self.y[idx]

    def split_even_odd(self, batch):
        return tuple(t[0::2] for t in batch), tuple(t[1::2] for t in batch)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def _evaluate(model, raw, X, A, y, bound):
    """new_bench ``evaluate_optimality`` columns at the frozen iterate, no duals."""
    model.eval()
    logits = model(X)
    f = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    c = raw(logits, A).reshape(-1).detach()
    grads = torch.autograd.grad(f, list(model.parameters()))
    model.train()
    rec = {"loss": float(f.detach()), "max_viol": float((c - bound).max()),
           "grad_norm": float(torch.sqrt(sum(g.pow(2).sum() for g in grads)))}
    rec.update({f"c_{j}": float(v) for j, v in enumerate(c)})
    return rec


def train(ghost_kwargs, seed, epochs, *, bound=BOUND, per_group=PER_GROUP):
    """Train one (config, seed); return the per-epoch history (new_bench opt.csv rows)."""
    X, A, y = load_data()
    n_groups = A.shape[1]
    raw, _ = fairness._build_raw_constraints("pairwise")

    # Reused for new_bench's epoch length and its inverse-propensity weights; nothing
    # iterates it.
    balanced = BalancedBatchSampler(group_onehot=A, batch_size=per_group * n_groups,
                                    drop_last=True)
    steps_per_epoch = len(balanced)
    group_weights = balanced.group_weights

    def objective_fn(net, batch):
        xb, ab, yb = batch
        return torch.nn.functional.binary_cross_entropy_with_logits(
            net(xb), yb, weight=(ab @ group_weights).unsqueeze(-1))

    def constraint_fn(net, batch):
        xb, ab, _ = batch
        return raw(net(xb), ab).reshape(-1) - bound

    set_seed(seed)
    # new_bench's 64-32-1 MLP.
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    cfg = GhostConfig(**ghost_kwargs)
    assert cfg.base_batch == per_group * n_groups, "base_batch must match per_group"
    solver = GhostSQP(model, objective_fn, constraint_fn, BalancedSampler(X, A, y),
                      cfg, rng=np.random.default_rng(seed))

    history = [{"epoch": 0, "time": 0.0, **_evaluate(model, raw, X, A, y, bound)}]
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        infos = [solver.step() for _ in range(steps_per_epoch)]
        elapsed = time.perf_counter() - started
        history.append({
            "epoch": epoch, "time": elapsed,
            **_evaluate(model, raw, X, A, y, bound),
            "gamma": infos[-1]["gamma"],
            "kappa": float(np.mean([i["kappa"] for i in infos])),
            "step_norm": float(np.mean([i["step_norm"] for i in infos])),
        })
        h = history[-1]
        print(f"  epoch {epoch:3d}  f {h['loss']:.4f}  max_viol {h['max_viol']:+.4f}  "
              f"|grad f| {h['grad_norm']:.3e}  kappa {h['kappa']:.3e}  "
              f"({elapsed / steps_per_epoch * 1e3:.0f} ms/step, {steps_per_epoch} steps)")
    return history


def tail_summary(history, tail=TAIL):
    """Tail means over the last ``tail`` epochs, as new_bench's select_best --tail."""
    window = history[-min(tail, len(history) - 1):]

    def mean(key):
        return float(np.mean([h[key] for h in window]))

    return {"tail epochs": len(window), "loss": mean("loss"),
            "max_viol": mean("max_viol"), "grad_norm": mean("grad_norm"),
            "epoch-0 loss": history[0]["loss"],
            "s/epoch": float(np.mean([h["time"] for h in history[1:]]))}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(description="E2d: Ghost SQP on Dutch")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    epochs = 1 if args.quick else args.epochs
    seeds = SEEDS[:1] if args.quick else SEEDS
    ghost_kwargs = OmegaConf.to_container(OmegaConf.load(ALGORITHM_CONF / "ghost.yaml").ghost)

    trajectories, rows = [], []
    for seed in seeds:
        print(f"\n=== Ghost on dutch_pairwise, seed {seed} ===")
        history = train(ghost_kwargs, seed, epochs)
        trajectories += [{"seed": seed, **h} for h in history]
        rows.append({"seed": seed, **tail_summary(history)})

    write_csv(trajectories, "e2d_opt", EXPERIMENT)
    summary = {"seeds": len(rows), **{k: float(np.mean([r[k] for r in rows]))
                                      for k in rows[0] if k != "seed"}}
    write_table([*rows, {"seed": "mean", **summary}], "e2d_summary", EXPERIMENT,
                title="E2d: Ghost on dutch_pairwise, tail-mean full-batch metrics")
    make_figure(trajectories)

    checks = Checks(enabled=args.check)
    checks.expect(summary["max_viol"] <= 0,
                  "P1: Ghost ends feasible on the full data",
                  f"tail-mean max_viol {summary['max_viol']:+.4f}")
    checks.expect(summary["loss"] < summary["epoch-0 loss"],
                  "P2: Ghost's tail-mean f is below its epoch-0 value",
                  f"{summary['epoch-0 loss']:.4f} -> {summary['loss']:.4f}")
    main_exit(checks, EXPERIMENT, "e2d_predictions")


def make_figure(trajectories):
    fig, axes, plt = figure(1, 2, row_height=2.2, sharex=True)
    for seed in dict.fromkeys(t["seed"] for t in trajectories):
        series = [t for t in trajectories if t["seed"] == seed]
        epochs = [t["epoch"] for t in series]
        axes[0].plot(epochs, [t["max_viol"] for t in series], lw=1.0, label=f"seed {seed}")
        axes[1].plot(epochs, [t["loss"] for t in series], lw=1.0)
    axes[0].axhline(0.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel("max-violation")
    axes[1].set_ylabel("objective f")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "e2d_trajectories", EXPERIMENT)
    plt.close(fig)


if __name__ == "__main__":
    main()
