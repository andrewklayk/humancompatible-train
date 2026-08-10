"""
E0d — data-parallel equivalence for the dual optimizers.

Prerequisite for E3, and the cheapest place to catch a bug class that is easy to
ship and hard to notice: with ``process_group`` set, a dual optimizer averages the
constraint vector across ranks before updating its multipliers, so *every replica
must hold the same duals* and ``G`` ranks at per-rank batch ``B`` must behave like
one rank at batch ``G*B``. Both are asserted here on the gloo CPU backend, so this
runs in CI without a GPU.

The second claim is where the interesting limitation lives, and it is stated
precisely rather than glossed:

* ``ReduceOp.AVG`` of per-rank means equals the mean over the union of the ranks'
  samples, and DDP averages parameter gradients, so a surrogate that is **linear
  in the constraint vector** — ``f + y'c``, i.e. ``ALM(rho=0)`` and ``nuPI(rho=0)``
  — is exactly equivalent, to summation order.
* A surrogate with a **quadratic term** is not: ``mean_r ||[c_r]_+||^2`` is not
  ``||[mean_r c_r]_+||^2``. The dual update is still exact (it sees the reduced
  vector), but the primal gradient differs by a variance term. ``ALM(rho>0)``,
  ``iALM`` and ``PBM`` (whose ``sum_i y_i p_i phi(c_i/p_i)`` is likewise nonlinear)
  are therefore only approximately equivalent, and by how much is measured.
* Independently of the surrogate, a **ratio-type** constraint such as
  ``E[p|a=1]/E[p|a=0] - 1 - tau`` is a nonlinear functional of expectations, so the
  average of per-rank values is not the global value however the surrogate is
  built. The gap is measured as a function of per-rank batch size, because that is
  what bounds how small a per-rank batch E2 and E3 may use.

Claims, registered before running:

D1 After ``k`` steps every rank holds bitwise-identical duals, for every method
   and both constraint types. Anything else means the reduction is wrong.
D2 For a mean-type constraint and a surrogate linear in ``c``, ``G x B`` matches
   ``1 x (G*B)`` in both duals and parameters to 1e-12 relative (not bitwise: the
   two group the same additions differently).
D3 For a mean-type constraint and a surrogate with a quadratic term, **one** step
   leaves the duals exactly equal — the dual update only ever sees the reduced
   vector — while the parameters already differ. Over ``k`` steps that primal
   discrepancy feeds back through the constraint values, so the duals diverge too.
   (D3 was first registered as "the duals still match after ``k`` steps"; the
   experiment falsified it, and the single-step split above is what is actually
   true. Both measurements are in the results table.)
D4 For a ratio-type constraint the **parameter** gap is nonzero for every method
   and shrinks as the per-rank batch grows, since the per-rank ratio estimate
   converges to the pooled one. The dual gap is reported too but is not the right
   statistic: a dual clamped to its safeguarding bound in both runs agrees for a
   reason that has nothing to do with the reduction, which is why the table also
   records whether a dual sits at its bound.
D5 Duals round-trip through ``state_dict`` identically on every rank.

Usage::

    python paper/e0/d_distributed.py --ranks 2      # re-execs itself under torchrun
    python paper/e0/d_distributed.py --quick
    python paper/e0/d_distributed.py --check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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

EXPERIMENT = "e0d"

STEPS = 20
PRIMAL_LR = 0.05
FEATURES = 8
TAU = 0.1
BATCH_SIZES = [16, 64, 256]
DATA_SEED = 7
MODEL_SEED = 11


# --------------------------------------------------------------------------- #
# problem: logistic regression with one expectation constraint
# --------------------------------------------------------------------------- #


def make_data(total: int):
    """Fixed dataset; rank ``r`` takes the contiguous slice of size ``total/G``."""
    generator = torch.Generator().manual_seed(DATA_SEED)
    dtype = torch.get_default_dtype()
    X = torch.randn(total, FEATURES, generator=generator, dtype=dtype)
    weights = torch.randn(FEATURES, generator=generator, dtype=dtype)
    noise = 0.3 * torch.randn(total, generator=generator, dtype=dtype)
    y = ((X @ weights + noise) > 0).to(dtype)
    # A sensitive attribute correlated with the label, so the group means differ
    # and the ratio constraint is actually binding.
    a = ((X[:, 0] + 0.5 * noise) > 0).to(dtype)
    return X, y, a


def make_model():
    torch.manual_seed(MODEL_SEED)
    return nn.Linear(FEATURES, 1)


def _predict(model, X):
    return torch.sigmoid(model(X).squeeze(-1))


def objective(model, X, y):
    p = _predict(model, X)
    return nn.functional.binary_cross_entropy(p, y)


def constraints_mean(model, X, y, a):
    """Mean-type: ``E[(p - y)^2] - tau <= 0``. Linear in the sample average."""
    p = _predict(model, X)
    return (((p - y) ** 2).mean() - TAU).reshape(1)


def constraints_ratio(model, X, y, a):
    """Ratio-type: ``E[p|a=1]/E[p|a=0] - 1 - tau <= 0``. A nonlinear functional."""
    p = _predict(model, X)
    high = p[a > 0.5].mean()
    low = p[a <= 0.5].mean()
    return (high / low - 1.0 - TAU).reshape(1)


CONSTRAINTS = {"mean": constraints_mean, "ratio": constraints_ratio}


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

# "linear" marks the surrogates that are linear in the constraint vector, which is
# exactly the property D2 needs and D3 denies.
METHODS = {
    "ALM (rho=0)": (lambda m, pg: ALM(m=m, lr=0.1, penalty=0.0, init_duals=0.5,
                                      is_ineq=True, process_group=pg), True),
    "ALM (rho=1)": (lambda m, pg: ALM(m=m, lr=0.1, penalty=1.0, init_duals=0.5,
                                      is_ineq=True, process_group=pg), False),
    "nuPI": (lambda m, pg: nuPI(m=m, ki=0.1, kp=0.1, nu=0.01, penalty=0.0,
                                init_duals=0.5, is_ineq=True, process_group=pg), True),
    "iALM": (lambda m, pg: iALM(m=m, beta=1.0, sigma=1.0, gamma=1.0, init_duals=0.5,
                                is_ineq=True, process_group=pg), False),
    "PBM": (lambda m, pg: PBM(m=m, gamma=0.5, penalty_mult=0.1, delta=1.0,
                              penalty_update="dimin_adapt", gamma_annealing=False,
                              penalty_annealing=False, process_group=pg), False),
}


def run(method_label, constraint_kind, X, y, a, steps, process_group=None,
        distributed=False):
    """Run ``steps`` constrained steps; return the final parameters and duals."""
    set_seed(0)
    model = make_model()
    if distributed:
        model = nn.parallel.DistributedDataParallel(model)
    build, _ = METHODS[method_label]
    dual = build(1, process_group)
    primal = torch.optim.SGD(model.parameters(), lr=PRIMAL_LR)

    constraint_fn = CONSTRAINTS[constraint_kind]
    for _ in range(steps):
        loss = objective(model, X, y)
        c = constraint_fn(model, X, y, a)
        primal.zero_grad()
        # The surrogate is built from local constraint values so autograd sees
        # this rank's dependence on the parameters; update() reduces them.
        dual.forward(loss, c).backward()
        primal.step()
        dual.update(c)

    inner = model.module if distributed else model
    group = dual.param_groups[0]
    bounds = (group.get("lower_bound"), group.get("upper_bound"))
    duals = dual.duals.detach().clone()
    at_bound = any(
        bound is not None and bool((duals == bound).any()) for bound in bounds
    )
    return {
        "params": torch.cat([p.detach().reshape(-1) for p in inner.parameters()]),
        "duals": duals,
        # params[0] specifically: PBM's group also carries its penalties in
        # params[1], so a blanket cat would compare a length-2 vector to a
        # length-1 one and fail for reasons unrelated to checkpointing.
        "state_dict_duals": dual.state_dict()["param_groups"][0]["params"][0]
                                .detach().clone(),
        "duals_at_bound": at_bound,
    }


# --------------------------------------------------------------------------- #
# worker: runs under torchrun
# --------------------------------------------------------------------------- #


def worker(outdir: Path, step_counts: list[int], batch_sizes: list[int]) -> None:
    use_float64()
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    for batch in batch_sizes:
        X, y, a = make_data(world * batch)
        lo, hi = rank * batch, (rank + 1) * batch
        X_local, y_local, a_local = X[lo:hi], y[lo:hi], a[lo:hi]
        for steps in step_counts:
            for kind in CONSTRAINTS:
                for label in METHODS:
                    result = run(label, kind, X_local, y_local, a_local, steps,
                                 process_group=dist.group.WORLD, distributed=True)
                    # Gather every rank's duals so the cross-rank check does not
                    # depend on the driver reading files in any particular order.
                    gathered = [torch.zeros_like(result["duals"])
                                for _ in range(world)]
                    dist.all_gather(gathered, result["duals"])
                    result["duals_all_ranks"] = torch.stack(gathered)
                    torch.save(result, outdir / _name(batch, steps, kind, label, rank))
    dist.barrier()
    dist.destroy_process_group()


def _slug(label: str) -> str:
    return label.replace(" ", "").replace("(", "_").replace(")", "").replace("=", "")


def _name(batch, steps, kind, label, rank):
    return f"b{batch}_s{steps}_{kind}_{_slug(label)}_r{rank}.pt"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = max(1e-30, float(b.abs().max()))
    return float((a - b).abs().max()) / scale


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="E0d: data-parallel equivalence")
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--outdir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    steps = 5 if args.quick else args.steps
    # One step isolates the effect of the surrogate's nonlinearity; `steps` shows
    # how it propagates into the duals through the constraint values.
    step_counts = [1, steps]
    batch_sizes = BATCH_SIZES[:1] if args.quick else BATCH_SIZES

    if args.worker:
        worker(Path(args.outdir), step_counts, batch_sizes)
        return

    use_float64()
    outdir = Path(args.outdir) if args.outdir else _default_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("*.pt"):
        stale.unlink()

    # Re-exec under torchrun. Doing it here rather than asking the user to means
    # the script is the single entry point the plan calls for.
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={args.ranks}",
        str(Path(__file__).resolve()),
        "--worker", "--outdir", str(outdir), "--steps", str(steps),
    ] + (["--quick"] if args.quick else [])
    print(f"launching {args.ranks} gloo ranks ...")
    environment = dict(os.environ, OMP_NUM_THREADS="1")
    completed = subprocess.run(command, env=environment)
    if completed.returncode != 0:
        raise SystemExit(f"torchrun failed with code {completed.returncode}")

    rows = []
    for batch in batch_sizes:
        # Single-process reference at the pooled batch size.
        X, y, a = make_data(args.ranks * batch)
        for step_count in step_counts:
            for kind in CONSTRAINTS:
                for label, (_, is_linear) in METHODS.items():
                    reference = run(label, kind, X, y, a, step_count)
                    per_rank = [
                        torch.load(outdir / _name(batch, step_count, kind, label, r),
                                   weights_only=False)
                        for r in range(args.ranks)
                    ]
                    duals_all = per_rank[0]["duals_all_ranks"]
                    rows.append({
                        "per-rank batch": batch,
                        "pooled batch": args.ranks * batch,
                        "steps": step_count,
                        "constraint": kind,
                        "method": label,
                        "surrogate linear in c": is_linear,
                        # D1: every rank against rank 0, bitwise.
                        "duals identical across ranks": all(
                            torch.equal(duals_all[0], duals_all[r])
                            for r in range(args.ranks)
                        ),
                        # D2 / D3 / D4.
                        "dual gap vs 1x pooled": _relative(per_rank[0]["duals"],
                                                           reference["duals"]),
                        "param gap vs 1x pooled": _relative(per_rank[0]["params"],
                                                            reference["params"]),
                        # A dual sitting on its safeguarding bound in both runs
                        # agrees for a reason unrelated to the reduction, so the
                        # dual gap above must be read next to this column.
                        "dual at bound": per_rank[0]["duals_at_bound"]
                                         or reference["duals_at_bound"],
                        # D5.
                        "state_dict round-trips": all(
                            torch.equal(r["duals"], r["state_dict_duals"])
                            for r in per_rank
                        ),
                    })
                    print(f"  b={batch:<4d} k={step_count:<3d} {kind:<6} {label:<12} "
                          f"same duals={rows[-1]['duals identical across ranks']!s:<5} "
                          f"dual gap={rows[-1]['dual gap vs 1x pooled']:.2e} "
                          f"param gap={rows[-1]['param gap vs 1x pooled']:.2e}"
                          + ("  (dual at bound)" if rows[-1]["dual at bound"] else ""))

    write_csv(rows, "e0d_equivalence", EXPERIMENT)
    write_table(
        rows, "e0d_equivalence", EXPERIMENT,
        title=f"E0d: {args.ranks} gloo ranks vs one process at the pooled batch, "
              f"after {steps} steps",
    )
    make_figure(rows, batch_sizes)

    checks = Checks(enabled=args.check)
    register_predictions(checks, rows, batch_sizes)
    main_exit(checks)


def _default_outdir() -> Path:
    """Scratch space for the per-rank tensors — not under ``paper/results``, which
    is committed."""
    import tempfile

    return Path(tempfile.gettempdir()) / "hc_train_e0d_ranks"


# --------------------------------------------------------------------------- #
# figure and predictions
# --------------------------------------------------------------------------- #


def make_figure(rows: list[dict], batch_sizes: list[int]):
    """The one thing worth a figure: how the ratio-type gap shrinks with batch size."""
    if len(batch_sizes) < 2:
        return
    many_steps = max(r["steps"] for r in rows)
    fig, axes, plt = figure(1, 2, row_height=2.2)
    for ax, kind in zip(axes, CONSTRAINTS):
        for label in METHODS:
            series = [r for r in rows if r["constraint"] == kind
                      and r["method"] == label and r["steps"] == many_steps]
            series.sort(key=lambda r: r["per-rank batch"])
            ax.plot([r["per-rank batch"] for r in series],
                    [max(r["param gap vs 1x pooled"], 1e-17) for r in series],
                    marker="o", markersize=3, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("per-rank batch size")
        ax.set_title(f"{kind}-type constraint")
    axes[0].set_ylabel("relative parameter gap\nvs one process at pooled batch")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "e0d_equivalence", EXPERIMENT)
    plt.close(fig)


EXACT = 1e-12


def register_predictions(checks: Checks, rows: list[dict], batch_sizes: list[int]):
    # D1 and D5 hold for everything.
    checks.expect(
        all(r["duals identical across ranks"] for r in rows),
        "D1: duals are bitwise identical on every rank, for every method and "
        "both constraint types",
        f"{sum(1 for r in rows if not r['duals identical across ranks'])} "
        f"of {len(rows)} configurations disagree",
    )
    checks.expect(
        all(r["state_dict round-trips"] for r in rows),
        "D5: duals round-trip through state_dict on every rank",
    )

    mean_rows = [r for r in rows if r["constraint"] == "mean"]
    many_steps = max(r["steps"] for r in rows)

    # D2 — holds at every step count.
    for row in [r for r in mean_rows if r["surrogate linear in c"]]:
        checks.expect(
            row["dual gap vs 1x pooled"] <= EXACT
            and row["param gap vs 1x pooled"] <= EXACT,
            f"D2: {row['method']} on a mean-type constraint matches 1 x "
            f"{row['pooled batch']} after {row['steps']} step(s) "
            f"(surrogate linear in c)",
            f"dual {row['dual gap vs 1x pooled']:.2e}, "
            f"param {row['param gap vs 1x pooled']:.2e}",
        )

    # D3 — one step separates the two halves of the claim.
    quadratic_one = [r for r in mean_rows
                     if not r["surrogate linear in c"] and r["steps"] == 1]
    for row in quadratic_one:
        checks.expect(
            row["dual gap vs 1x pooled"] <= EXACT,
            f"D3: after one step {row['method']}'s duals match 1 x "
            f"{row['pooled batch']} exactly — the dual update only ever sees the "
            f"reduced constraint vector",
            f"dual {row['dual gap vs 1x pooled']:.2e}",
        )
    checks.expect(
        all(r["param gap vs 1x pooled"] > EXACT for r in quadratic_one),
        "D3: after one step the parameters already differ, because "
        "mean_r ||[c_r]_+||^2 != ||[mean_r c_r]_+||^2",
        "; ".join(f"{r['method']} {r['param gap vs 1x pooled']:.2e}"
                  for r in quadratic_one),
    )
    quadratic_many = [r for r in mean_rows
                      if not r["surrogate linear in c"] and r["steps"] == many_steps]
    checks.expect(
        all(r["dual gap vs 1x pooled"] > EXACT for r in quadratic_many),
        f"D3: over {many_steps} steps that primal discrepancy propagates into the "
        f"duals through the constraint values",
        "; ".join(f"{r['method']} {r['dual gap vs 1x pooled']:.2e}"
                  for r in quadratic_many),
    )

    # D4 — stated on the parameter gap, which no safeguarding clamp can zero out.
    ratio_rows = [r for r in rows if r["constraint"] == "ratio"]
    checks.expect(
        all(r["param gap vs 1x pooled"] > EXACT for r in ratio_rows),
        "D4: no method reproduces the pooled-batch parameters under a ratio-type "
        "constraint, because the average of per-rank ratios is not the pooled ratio",
        f"smallest param gap "
        f"{min(r['param gap vs 1x pooled'] for r in ratio_rows):.2e}",
    )
    if len(batch_sizes) >= 2:
        for label in METHODS:
            series = sorted(
                (r for r in ratio_rows
                 if r["method"] == label and r["steps"] == many_steps),
                key=lambda r: r["per-rank batch"],
            )
            gaps = [r["param gap vs 1x pooled"] for r in series]
            checks.expect(
                gaps[-1] < gaps[0],
                f"D4: {label}'s ratio-type gap shrinks as the per-rank batch grows",
                "; ".join(f"b={r['per-rank batch']}: {g:.2e}"
                          for r, g in zip(series, gaps)),
            )


if __name__ == "__main__":
    main()
