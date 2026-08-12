"""
E2b — data-parallel correctness on a constraint the reduction actually changes.

This is the claim E3 cannot make. E3's sparsity constraint is a closed form in the
gate parameters, so every rank computes the same value and ``all_reduce(c, AVG)``
averages identical numbers -- it cannot be caught being wrong. Here the constraint
is a *statistic of the data*, the per-rank values genuinely differ, and the
reduction has real work to do: turn G per-shard estimates into the pooled one.

The pivot is a fact about `fairret`'s ``PositiveRate`` that is easy to miss.
It is a ``LinearFractionalStatistic`` with ``denom_slope = 0``, i.e. per group it
is ``sum_i p_i 1[g_i = g] / sum_i 1[g_i = g]`` -- a *ratio of sums*, which E0d
found never reproduces a pooled run exactly. But under **balanced batching** the
denominator is exactly ``PER_GROUP`` on every rank, a constant, so the statistic
collapses to a plain mean and becomes linear in the sample. So the
`BalancedBatchSampler` that per-group constraints already require for statistical
reasons turns out to be the same thing that makes data-parallel equivalence exact.
Those two facts were previously unconnected.

Batches are constructed explicitly rather than drawn from a sampler, because the
whole experiment rests on one invariant -- *the union of the ranks' batches is the
single-process batch* -- and building it by hand makes that invariant obvious
instead of a property to be trusted about a shuffler. (It also means this script
does not need the distributed-aware sampler that the library still lacks; what it
shows is *why* that sampler has to shard per group.)

Claims, registered before running:

D1 After ``k`` steps every rank holds bitwise-identical duals, for every method and
   both batch modes. This is E0d's D1 on a real constraint.
D2 With **balanced** sharding and a surrogate **linear in c** (``ALM(rho=0)``,
   ``nuPI(rho=0)``), ``G x B`` matches ``1 x (G*B)`` to 1e-12 in duals and
   parameters -- the constraint is mean-type by the argument above, so the
   reduction is exact.
D3 With **balanced** sharding and a surrogate carrying a **quadratic** term
   (``ALM(rho=1)``, ``iALM``), the parameters differ after a single step --
   **but only where the penalty is live.** It acts on ``[c]_+``, so at a strictly
   feasible iterate its gradient is identically zero on every rank and a quadratic
   surrogate is exactly as reproducible as a linear one. E2b therefore runs two
   bounds, 0.05 (feasible from step 0) and -0.02 (violated from step 0), and states
   both halves. This refines E0d, which saw only the violated case, and it is the
   ``[c]_+`` fix of E0a showing up in a place nothing predicted it would.
   ``PBM`` is a *third* mechanism, split out as D3c: its barrier
   ``sum_i y_i p_i phi(c_i / p_i)`` is nonlinear in ``c`` at every ``c``, with no
   clamp to switch off, so its inexactness does not depend on the bound at all.
D4 With **shuffled** sharding, even the linear surrogate fails to match: the
   per-group counts differ across ranks, the denominator stops being constant, and
   ``PositiveRate`` reverts to a genuine ratio of sums. **This is the load-bearing
   result** -- it shows the reduction is doing something that can be got wrong, and
   names the precondition under which it is exact.
D5 The aggregate shape (``FairretAgg``, a norm over the per-group statistics) fails
   to match *regardless* of batch mode, because a norm of means is not the mean of
   norms. Balanced batching fixes the denominator, not the outer nonlinearity.
D6 Duals round-trip through ``state_dict`` identically on every rank.

Usage::

    python paper/e2/b_parallel.py --ranks 2     # re-execs itself under torchrun
    python paper/e2/b_parallel.py --quick
    python paper/e2/b_parallel.py --check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from humancompatible.train.dual_optim import ALM, PBM, iALM, nuPI
from paper._harness import (
    Checks,
    figure,
    main_exit,
    save_figure,
    set_seed,
    use_float64,
    write_csv,
    write_table,
)
from paper.problems import fairness

EXPERIMENT = "e2b"

STEPS = 10
PRIMAL_LR = 0.02
DUAL_LR = 0.05
PER_GROUP = 8           # per rank, per group, per step
SUBSAMPLE = 4000        # rows kept; this is about exactness, not statistics
MODEL_SEED = 11
BATCH_SEED = 5
EXACT = 1e-12

# ``linear`` marks the surrogates that are linear in the constraint vector -- the
# property D2 needs and D3 denies.
#
# Every builder takes ``(m, process_group)`` and MUST forward the group: without it
# no all-reduce happens, every rank drives its duals from purely local constraints,
# and D4 then "passes" for entirely the wrong reason -- nothing is being reduced,
# so of course nothing matches the pooled run. D1 is the guard that catches this.
METHODS = {
    "ALM (rho=0)": (lambda m, pg: ALM(m=m, lr=DUAL_LR, penalty=0.0, init_duals=0.5,
                                      is_ineq=True, process_group=pg), True),
    "nuPI (rho=0)": (lambda m, pg: nuPI(m=m, ki=DUAL_LR, kp=DUAL_LR, nu=0.01,
                                        penalty=0.0, init_duals=0.5, is_ineq=True,
                                        process_group=pg), True),
    "ALM (rho=1)": (lambda m, pg: ALM(m=m, lr=DUAL_LR, penalty=1.0, init_duals=0.5,
                                      is_ineq=True, process_group=pg), False),
    "iALM": (lambda m, pg: iALM(m=m, beta=1.0, sigma=1.0, gamma=1.0, init_duals=0.5,
                                is_ineq=True, process_group=pg), False),
    "PBM": (lambda m, pg: PBM(m=m, gamma=0.5, penalty_mult=0.1, delta=1.0,
                              penalty_update="dimin_adapt", gamma_annealing=False,
                              penalty_annealing=False, process_group=pg), False),
}

# Two fairness budgets, chosen to put the quadratic penalty on both sides of its
# own switch. The penalty acts on ``[c]_+``, so at 0.05 the untrained model is
# strictly feasible on every rank (all pairwise gaps start near zero), the penalty
# gradient is identically zero, and a quadratic surrogate is indistinguishable from
# a linear one. At -0.02 the same model is violated from step 0, so the penalty is
# live. D3 is stated conditionally on which regime a row is in.
BOUNDS = [0.05, -0.02]

SHAPES = ["pairwise", "agg"]
# Both modes draw the *same* pooled batch and differ only in how it is split
# across ranks, so the single-process reference is literally the same computation
# in both -- which makes D4 a controlled comparison of the sharding alone.
MODES = ["balanced", "shuffled"]

# Three distinct sources of data-parallel inexactness, and they are NOT the same
# argument. The clamped quadratic switches off at a feasible iterate; the barrier
# never does; the aggregate shape's outer norm never does either (D5).
CLAMPED_QUADRATIC = ["ALM (rho=1)", "iALM"]
BARRIER = ["PBM"]


# --------------------------------------------------------------------------- #
# the problem, subsampled, plus explicit batch construction
# --------------------------------------------------------------------------- #


def _load(shape, bound):
    """One E2 problem, subsampled and cast to the ambient dtype."""
    problem = fairness.build("income", shape, bound=bound, split_seed=0)
    features, sens, labels = problem.train
    keep = _stratified_head(sens, SUBSAMPLE)
    problem.train = (features[keep], sens[keep], labels[keep])
    return problem


def _stratified_head(sens, total):
    """First ``total / G`` rows of each group, so every group is present."""
    groups = sens.argmax(1)
    per_group = total // sens.shape[1]
    keep = []
    for group in range(sens.shape[1]):
        keep.append(torch.nonzero(groups == group).reshape(-1)[:per_group])
    return torch.cat(keep)


def batches(sens, *, mode, world, steps, per_group, seed):
    """Deterministic list of per-step index tensors, one entry per step.

    Each entry is the **pooled** batch: rank ``r`` takes
    ``batch[r * size : (r + 1) * size]``. Built so that the union over ranks is
    exactly the single-process batch, which is the invariant every claim here
    rests on.

    Both modes draw the **same pooled batch** -- ``world * per_group`` indices of
    every group -- and differ only in the order, hence in which rank gets which
    rows:

    * ``balanced`` lays them out rank-major, so every rank's contiguous slice holds
      exactly ``per_group`` of each group and the statistic's denominator is a
      constant.
    * ``shuffled`` permutes the pooled batch first, so the ranks' group counts
      differ. That is the single variable D4 is about, and holding the pooled batch
      fixed makes the comparison controlled rather than confounded with a different
      sample.

    A fully random draw from the whole dataset was the obvious alternative and is
    wrong: a rank can then receive *zero* members of a group, and
    ``PositiveRate``'s denominator is that group's count, so the statistic becomes
    a division by zero rather than an uneven estimate. Permuting a batch that
    already contains ``world * per_group`` of each group keeps every group present
    while still making the per-rank counts uneven.
    """
    generator = torch.Generator().manual_seed(seed)
    n_groups = sens.shape[1]
    groups = sens.argmax(1)
    by_group = [torch.nonzero(groups == g).reshape(-1) for g in range(n_groups)]

    out = []
    for _ in range(steps):
        per_rank = [[] for _ in range(world)]
        for indices in by_group:
            chosen = indices[torch.randperm(len(indices), generator=generator)
                             [: world * per_group]]
            for rank in range(world):
                per_rank[rank].append(chosen[rank * per_group:(rank + 1) * per_group])
        pooled = torch.cat([torch.cat(block) for block in per_rank])
        if mode == "balanced":
            out.append(pooled)
        elif mode == "shuffled":
            out.append(pooled[torch.randperm(len(pooled), generator=generator)])
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out


def make_model(n_features):
    torch.manual_seed(MODEL_SEED)
    return nn.Sequential(
        nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 1)
    )


def run(problem, method, index_batches, *, rank, world, process_group=None,
        distributed=False):
    """Run the given batches; return final parameters, duals and diagnostics."""
    set_seed(0)
    features, sens, labels = problem.train
    model = make_model(problem.n_features)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model)
    build, _ = METHODS[method]
    dual = build(problem.m, process_group)
    primal = torch.optim.SGD(model.parameters(), lr=PRIMAL_LR)

    # Whether the quadratic penalty was ever *live*. It acts on ``[c]_+``, so at a
    # strictly feasible iterate its gradient is zero on every rank and a quadratic
    # surrogate is indistinguishable from a linear one -- which is exactly the
    # condition D3 turns on.
    max_local_c = -float("inf")

    for pooled in index_batches:
        size = len(pooled) // world
        mine = pooled[rank * size:(rank + 1) * size] if distributed else pooled
        primal.zero_grad()
        logits = model(features[mine])
        objective = problem.objective(logits, labels[mine])
        constraints = problem.constraints(logits, sens[mine])
        max_local_c = max(max_local_c, float(constraints.detach().max()))
        # The surrogate is built from LOCAL constraint values so autograd sees this
        # rank's dependence on the parameters; forward_update reduces them.
        dual.forward_update(objective, constraints).backward()
        primal.step()

    inner = model.module if distributed else model
    group = dual.param_groups[0]
    duals = dual.duals.detach().clone()
    at_bound = any(
        bound is not None and bool((duals == bound).any())
        for bound in (group.get("lower_bound"), group.get("upper_bound"))
    )
    return {
        "params": torch.cat([p.detach().reshape(-1) for p in inner.parameters()]),
        "duals": duals,
        # params[0] specifically: PBM's group also carries its penalties in
        # params[1], so a blanket cat would compare vectors of different lengths.
        "state_dict_duals": dual.state_dict()["param_groups"][0]["params"][0]
                                .detach().clone(),
        "duals_at_bound": at_bound,
        "max_local_c": max_local_c,
    }


# --------------------------------------------------------------------------- #
# worker / driver
# --------------------------------------------------------------------------- #


def worker(outdir: Path, step_counts, shapes, modes, bounds):
    use_float64()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    for shape in shapes:
        for bound in bounds:
            problem = _load(shape, bound)
            for mode in modes:
                for steps in step_counts:
                    index_batches = batches(problem.train[1], mode=mode,
                                            world=world, steps=steps,
                                            per_group=PER_GROUP, seed=BATCH_SEED)
                    for method in METHODS:
                        result = run(problem, method, index_batches, rank=rank,
                                     world=world, process_group=dist.group.WORLD,
                                     distributed=True)
                        gathered = [torch.zeros_like(result["duals"])
                                    for _ in range(world)]
                        dist.all_gather(gathered, result["duals"])
                        result["duals_all_ranks"] = torch.stack(gathered)
                        torch.save(result, outdir / _name(shape, bound, mode,
                                                          steps, method, rank))
    dist.barrier()
    dist.destroy_process_group()


def _slug(label):
    return (label.replace(" ", "").replace("(", "_").replace(")", "")
            .replace("=", ""))


def _name(shape, bound, mode, steps, method, rank):
    return (f"{shape}_b{bound:g}_{mode}_s{steps}_{_slug(method)}_r{rank}.pt")


def _relative(a, b):
    scale = max(1e-30, float(b.abs().max()))
    return float((a - b).abs().max()) / scale


def main(argv=None):
    parser = argparse.ArgumentParser(description="E2b: data-parallel correctness")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--outdir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    steps = 3 if args.quick else args.steps
    step_counts = [1, steps]
    shapes = SHAPES[:1] if args.quick else SHAPES
    modes = MODES
    bounds = BOUNDS

    if args.worker:
        worker(Path(args.outdir), step_counts, shapes, modes, bounds)
        return

    use_float64()
    outdir = (Path(args.outdir) if args.outdir
              else Path(tempfile.gettempdir()) / "hc_train_e2b_ranks")
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.pt"):
        stale.unlink()

    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={args.ranks}",
        str(Path(__file__).resolve()),
        "--worker", "--outdir", str(outdir), "--steps", str(steps),
    ] + (["--quick"] if args.quick else [])
    print(f"launching {args.ranks} gloo ranks ...")
    completed = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if completed.returncode != 0:
        raise SystemExit(f"torchrun failed with code {completed.returncode}")

    rows = []
    for shape in shapes:
      for bound in bounds:
        problem = _load(shape, bound)
        for mode in modes:
            for step_count in step_counts:
                index_batches = batches(problem.train[1], mode=mode,
                                        world=args.ranks, steps=step_count,
                                        per_group=PER_GROUP, seed=BATCH_SEED)
                for method, (_, is_linear) in METHODS.items():
                    # Single process, pooled batch: world=1 and rank=0 make `mine`
                    # the whole pooled index set.
                    reference = run(problem, method, index_batches, rank=0, world=1)
                    per_rank = [
                        torch.load(outdir / _name(shape, bound, mode, step_count,
                                                  method, r), weights_only=False)
                        for r in range(args.ranks)
                    ]
                    duals_all = per_rank[0]["duals_all_ranks"]
                    rows.append({
                        "constraint": shape,
                        "m": problem.m,
                        "bound": bound,
                        "sharding": mode,
                        "steps": step_count,
                        "method": method,
                        "surrogate linear in c": is_linear,
                        # Whether the quadratic penalty was ever live: it acts on
                        # [c]_+, so a strictly feasible iterate zeroes its gradient
                        # on every rank and it cannot create any discrepancy.
                        "penalty live": max(per_rank[0]["max_local_c"],
                                            reference["max_local_c"]) > 0,
                        "duals identical across ranks": all(
                            torch.equal(duals_all[0], duals_all[r])
                            for r in range(args.ranks)
                        ),
                        "dual gap vs 1x pooled": _relative(per_rank[0]["duals"],
                                                           reference["duals"]),
                        "param gap vs 1x pooled": _relative(per_rank[0]["params"],
                                                            reference["params"]),
                        "dual at bound": per_rank[0]["duals_at_bound"]
                                         or reference["duals_at_bound"],
                        "state_dict round-trips": all(
                            torch.equal(r["duals"], r["state_dict_duals"])
                            for r in per_rank
                        ),
                    })
                    row = rows[-1]
                    print(f"  {shape:<9} b={bound:<6g} {mode:<9} "
                          f"k={step_count:<3} {method:<13} "
                          f"same duals={row['duals identical across ranks']!s:<5} "
                          f"dual gap={row['dual gap vs 1x pooled']:.2e} "
                          f"param gap={row['param gap vs 1x pooled']:.2e}")

    write_csv(rows, "e2b_equivalence", EXPERIMENT)
    write_table(rows, "e2b_equivalence", EXPERIMENT,
                title=f"E2b: {args.ranks} gloo ranks vs one process at the pooled "
                      f"batch, after {steps} steps")
    make_figure(rows)

    checks = Checks(enabled=args.check)
    register_predictions(checks, rows)
    main_exit(checks, EXPERIMENT, "e2b_predictions")


def make_figure(rows):
    """The one thing worth a picture: balanced sharding is exact for a linear
    surrogate, shuffled sharding is not, and the aggregate shape is not either way."""
    shapes = list(dict.fromkeys(r["constraint"] for r in rows))
    many = max(r["steps"] for r in rows)
    fig, axes, plt = figure(1, len(shapes), row_height=2.4)
    width = 0.35
    for ax, shape in zip(axes, shapes):
        methods = list(METHODS)
        positions = np.arange(len(methods))
        for offset, mode in enumerate(MODES):
            values = []
            for method in methods:
                match = [r for r in rows if r["constraint"] == shape
                         and r["sharding"] == mode and r["method"] == method
                         and r["steps"] == many]
                values.append(max(match[0]["param gap vs 1x pooled"], 1e-17)
                              if match else 1e-17)
            ax.bar(positions + offset * width, values, width, label=mode)
        ax.axhline(EXACT, color="k", lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xticks(positions + width / 2)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_title(f"{shape} (m={[r['m'] for r in rows if r['constraint']==shape][0]})")
    axes[0].set_ylabel("relative parameter gap\nvs one process at pooled batch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, "e2b_equivalence", EXPERIMENT)
    plt.close(fig)


def register_predictions(checks, rows):
    def pick(**criteria):
        return [r for r in rows
                if all(r[key] == value for key, value in criteria.items())]

    # D1 / D6 hold for everything.
    disagree = [r for r in rows if not r["duals identical across ranks"]]
    checks.expect(
        not disagree,
        "D1: duals are bitwise identical on every rank, for every method, "
        "constraint shape and sharding mode",
        f"{len(disagree)} of {len(rows)} configurations disagree",
    )
    checks.expect(
        all(r["state_dict round-trips"] for r in rows),
        "D6: duals round-trip through state_dict on every rank",
    )

    linear_balanced = pick(constraint="pairwise", sharding="balanced",
                           **{"surrogate linear in c": True})
    # D2 — exact, at every step count.
    for row in linear_balanced:
        checks.expect(
            row["dual gap vs 1x pooled"] <= EXACT
            and row["param gap vs 1x pooled"] <= EXACT,
            f"D2: {row['method']} on the pairwise constraint with balanced "
            f"sharding matches 1 x pooled after {row['steps']} step(s) — the "
            f"per-group denominator is constant, so the statistic is mean-type",
            f"dual {row['dual gap vs 1x pooled']:.2e}, "
            f"param {row['param gap vs 1x pooled']:.2e}",
        )

    # D3 — the quadratic term breaks exactness, but ONLY where it is live. It acts
    # on ``[c]_+``, so at a strictly feasible iterate its gradient is identically
    # zero on every rank and a quadratic surrogate is indistinguishable from a
    # linear one. That is why E2b runs two bounds: 0.05 (feasible from step 0) and
    # -0.02 (violated from step 0). Splitting on `penalty live` turns what would
    # otherwise be a claim that silently depends on the bound into a sharp
    # conditional -- and it connects two E0-era facts, since the clamp responsible
    # IS the [c]_+ fix that E0a's fixed-point test was built to catch.
    nonlinear = pick(constraint="pairwise", sharding="balanced", steps=1,
                     **{"surrogate linear in c": False})
    # The clamp argument applies only to the methods whose nonlinearity *is* the
    # [c]_+ quadratic. PBM's is a barrier, handled separately below.
    clamped = [r for r in nonlinear if r["method"] in CLAMPED_QUADRATIC]
    live = [r for r in clamped if r["penalty live"]]
    inert = [r for r in clamped if not r["penalty live"]]

    checks.expect(
        bool(live) and all(r["param gap vs 1x pooled"] > EXACT for r in live),
        "D3a: where the quadratic penalty is LIVE (some c_r > 0), one step already "
        "makes the parameters differ, because "
        "mean_r ||[c_r]_+||^2 != ||[mean_r c_r]_+||^2",
        "; ".join(f"{r['method']} b={r['bound']:g} "
                  f"{r['param gap vs 1x pooled']:.2e}" for r in live) or "no live rows",
    )
    checks.expect(
        bool(inert) and all(r["param gap vs 1x pooled"] <= EXACT for r in inert),
        "D3b: where it is INERT (every c_r <= 0), the same surrogate is exactly as "
        "reproducible as a linear one — [c]_+ zeroes its gradient on every rank, so "
        "there is no variance term left to disagree about",
        "; ".join(f"{r['method']} b={r['bound']:g} "
                  f"{r['param gap vs 1x pooled']:.2e}" for r in inert) or "no inert rows",
    )
    # D3c — PBM is a third mechanism, and the [c]_+ argument does not reach it.
    barrier = [r for r in nonlinear if r["method"] in BARRIER]
    checks.expect(
        bool(barrier) and all(r["param gap vs 1x pooled"] > EXACT for r in barrier),
        "D3c: PBM never matches, in EITHER regime — its surrogate is a barrier "
        "sum_i y_i p_i phi(c_i / p_i), which is nonlinear in c at every c. There is "
        "no clamp to switch off when the iterate is feasible, so unlike the "
        "quadratic methods its inexactness does not depend on the bound",
        "; ".join(f"{r['method']} b={r['bound']:g} live={r['penalty live']} "
                  f"{r['param gap vs 1x pooled']:.2e}" for r in barrier),
    )
    checks.expect(
        all(r["dual gap vs 1x pooled"] <= EXACT for r in nonlinear),
        "D3d: in every regime the duals still match exactly after one step, because "
        "the dual update only ever sees the reduced constraint vector",
        "; ".join(f"{r['method']} b={r['bound']:g} "
                  f"{r['dual gap vs 1x pooled']:.2e}" for r in nonlinear),
    )

    # D4 — the load-bearing one.
    linear_shuffled = pick(constraint="pairwise", sharding="shuffled",
                         **{"surrogate linear in c": True})
    checks.expect(
        all(r["param gap vs 1x pooled"] > EXACT for r in linear_shuffled),
        "D4: with shuffled sharding even a linear surrogate stops matching — the "
        "per-group counts differ across ranks, so PositiveRate reverts to a "
        "genuine ratio of sums. The reduction is load-bearing, and balanced "
        "sharding is its precondition",
        "; ".join(f"{r['method']} k={r['steps']} "
                  f"{r['param gap vs 1x pooled']:.2e}" for r in linear_shuffled),
    )

    # D5 — the aggregate shape's outer nonlinearity survives balancing.
    for mode in MODES:
        agg = pick(constraint="agg", sharding=mode,
                   **{"surrogate linear in c": True})
        if not agg:
            continue
        checks.expect(
            all(r["param gap vs 1x pooled"] > EXACT for r in agg),
            f"D5: the aggregate shape does not match under {mode} sharding either "
            f"— a norm of per-group means is not the mean of norms, and balanced "
            f"batching fixes the denominator, not the outer nonlinearity",
            "; ".join(f"{r['method']} k={r['steps']} "
                      f"{r['param gap vs 1x pooled']:.2e}" for r in agg),
        )


if __name__ == "__main__":
    main()
