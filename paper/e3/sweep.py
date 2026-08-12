"""
E3 — do the dual methods deliver a requested sparsity on a large LM, and at what cost?

Sweeps every constrained method in :mod:`paper.e3.run_llm` against three target densities,
plus the fixed-penalty formulation of arXiv:2208.04425 Eq. (1)-(2) as the baseline their
Fig. 1 compares against. Three numbers per run:

* ``achieved - eps`` — did the method land on the density that was asked for;
* ``eval_ppl`` — what the model is worth there, at the median (test-time) gates on
  held-out blocks;
* ``tokens_per_s`` and the step-time breakdown — what the dual layer cost to get it.

The headline is the **final iterate**, not an average over a tail of steps. E2a averages,
because its constraint is a sample mean whose sign flickers with the minibatch; here the
constraint is a closed form in the gate parameters, with no data and no sampling noise in
it, so the last value *is* the achieved density. The full trajectory is written anyway.

**Distributed by necessity, not as a subject.** At 0.5 B parameters the grid does not run
on one GPU. Every rank walks the same list of configurations at full world size and rank 0
writes; the model is dropped between configurations because 21 of them in one process
otherwise fragments the allocator.

**No predictions are registered** — see ``run_llm.py``'s docstring. ``--check`` is accepted
and exits 0.

Usage::

    python paper/e3/sweep.py --quick                 # CPU stub, seconds
    python paper/e3/sweep.py --quick --ranks 2       # same, 2 gloo ranks
    python paper/e3/sweep.py --model Qwen/Qwen2.5-0.5B --tokens .../tokens.bin --ranks 4
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from paper._harness import figure, save_figure, write_csv, write_table
from paper.e3 import run_llm

EXPERIMENT = "e3"

EPS = (0.3, 0.5, 0.7)
# Their Eq. (1)-(2): the loss plus a fixed multiple of the expected L0. One coefficient
# per run, no target -- which is exactly the difficulty the constrained form removes.
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0)


def _configs(args):
    """The grid, as (label, overrides) pairs."""
    out = []
    for method in run_llm.CONSTRAINED:
        for eps in args.eps:
            out.append((f"{method}@{eps:g}", {"method": method, "eps": eps}))
    for lam in args.lambdas:
        out.append((f"penalty@{lam:g}", {"method": "penalty", "penalty": lam}))
    return out


def _make_args(args, overrides):
    """A single-run namespace: run_llm's defaults, this driver's flags, the overrides."""
    one = run_llm.build_parser().parse_args([])
    for key in ("model", "tokens", "synthetic", "vocab_size", "granularity", "steps",
                "seq_len", "batch_size", "lr", "gate_lr", "dual_lr", "dtype", "seed",
                "quick",
                "eval_batches"):
        setattr(one, key, getattr(args, key))
    for key, value in overrides.items():
        setattr(one, key, value)
    return run_llm.finalize(one)


def run_grid(args, *, rank, world, device, process_group):
    runs, trajectories = [], []
    configs = _configs(args)
    for index, (label, overrides) in enumerate(configs, start=1):
        if rank == 0:
            print(f"\n[{index}/{len(configs)}] {label}", flush=True)
        result = run_llm.train(
            _make_args(args, overrides), rank=rank, world=world,
            device=device, process_group=process_group,
        )
        if rank == 0:
            summary = dict(result["summary"])
            summary["label"] = label
            summary["penalty"] = overrides.get("penalty", "")
            if summary["method"] == "penalty":
                # There is no target in the penalized formulation; leaving run_llm's
                # default eps in the row would read as one.
                summary["eps"] = ""
            runs.append(summary)
            for row in result["rows"]:
                trajectories.append({"label": label, **row})
        # The next configuration builds a fresh model; without this the previous one's
        # activations and optimizer state stay resident until the allocator is pressured.
        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return runs, trajectories


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def _summaries(runs):
    """The headline table, and the penalty baseline beside it."""
    constrained, penalty = [], []
    for run in runs:
        if run["method"] == "penalty":
            penalty.append({
                "lambda": run["penalty"],
                "achieved density": run["final_density_mean"],
                **{f"distance to eps={e:g}": run["final_density_mean"] - e for e in EPS},
                "eval ppl": run.get("eval_ppl", float("nan")),
                "tokens/s": run["tokens_per_s"],
            })
            continue
        constrained.append({
            "method": run["method"],
            "eps": run["eps"],
            "achieved density": run["final_density_mean"],
            "achieved - eps": run["final_density_mean"] - run["eps"],
            "per-layer min": run["final_density_min"],
            "per-layer max": run["final_density_max"],
            "eval ppl": run.get("eval_ppl", float("nan")),
            "tokens/s": run["tokens_per_s"],
            "ms/step": run["ms_per_step"],
            "ms dual": run["ms_dual"],
        })
    constrained.sort(key=lambda r: (r["eps"], r["method"]))
    penalty.sort(key=lambda r: r["lambda"])
    return constrained, penalty


def _figures(runs, trajectories, eps_values):
    by_label = {}
    for row in trajectories:
        by_label.setdefault(row["label"], []).append(row)

    # One panel per target: every method's density trajectory against the line it was
    # asked to reach.
    fig, axes, plt = figure(1, len(eps_values), row_height=2.4)
    for ax, eps in zip(axes, eps_values):
        for label, rows in by_label.items():
            if not label.endswith(f"@{eps:g}") or label.startswith("penalty"):
                continue
            ax.plot([r["step"] for r in rows], [r["density_mean"] for r in rows],
                    lw=1.0, label=label.split("@")[0])
        ax.axhline(eps, color="k", ls="--", lw=0.8)
        ax.set_title(f"$\\varepsilon$ = {eps:g}")
        ax.set_xlabel("step")
    axes[0].set_ylabel("expected density")
    axes[-1].legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "e3_sweep_trajectories", EXPERIMENT)
    plt.close(fig)

    # Achieved against requested, with the penalty sweep as a horizontal spread: it has
    # no target to be plotted against, which is the point being made.
    fig, axes, plt = figure(1, 1, row_height=2.6)
    ax = axes[0]
    methods = sorted({r["method"] for r in runs if r["method"] != "penalty"})
    for method in methods:
        points = sorted((r["eps"], r["final_density_mean"])
                        for r in runs if r["method"] == method)
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", ms=3, lw=1.0, label=method)
    for run in runs:
        if run["method"] == "penalty":
            ax.axhline(run["final_density_mean"], color="grey", lw=0.6, alpha=0.6)
    lo, hi = min(eps_values), max(eps_values)
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8, label="target")
    ax.set_xlabel("requested density $\\varepsilon$")
    ax.set_ylabel("achieved density")
    ax.set_title("grey: fixed-penalty runs (no target to request)")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "e3_sweep_achieved", EXPERIMENT)
    plt.close(fig)


def write_artifacts(runs, trajectories, eps_values):
    write_csv(trajectories, "e3_sweep_trajectories", EXPERIMENT)
    write_csv(runs, "e3_sweep_runs", EXPERIMENT)
    constrained, penalty = _summaries(runs)
    write_table(constrained, "e3_sweep", EXPERIMENT,
                title="E3: achieved density, held-out perplexity at the median gates, "
                      "and throughput (final iterate)")
    if penalty:
        write_table(penalty, "e3_penalty", EXPERIMENT,
                    title="E3: the fixed-penalty baseline, one coefficient per run")
    _figures(runs, trajectories, eps_values)
    print()
    for row in constrained:
        print(f"  {row['method']:<17} eps={row['eps']:.2f}  "
              f"achieved {row['achieved density']:.4f} ({row['achieved - eps']:+.4f})  "
              f"ppl {row['eval ppl']:.2f}  {row['tokens/s']:.0f} tok/s")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub")
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--granularity", default="layer")
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS))
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8, help="per rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--dual-lr", type=float, default=1e-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=None,
                        help="ranks to launch; default is every visible GPU")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="accepted for parity; E3 registers no predictions")
    args = parser.parse_args(argv)

    if args.quick:
        args.eps = args.eps[:2]
        args.lambdas = args.lambdas[:2]
        args.steps = 6

    rank, world, local_rank = run_llm._dist_env()

    if rank >= 0:  # a worker under torchrun
        device = run_llm._pick_device(local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        try:
            runs, trajectories = run_grid(
                args, rank=rank, world=world, device=device,
                process_group=dist.group.WORLD,
            )
            if rank == 0:
                write_artifacts(runs, trajectories, args.eps)
        finally:
            dist.barrier()
            dist.destroy_process_group()
        return

    ranks = args.ranks if args.ranks is not None else max(1, torch.cuda.device_count())
    if ranks > 1:
        # The shard must exist before the ranks race for it; run_llm writes it only in
        # its own single-process path.
        run_llm._ensure_tokens(_make_args(args, {"method": "adam"}))
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={ranks}", str(Path(__file__).resolve()),
        ] + _rebuild_argv(args)
        print(f"launching {ranks} ranks ...", flush=True)
        completed = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1"))
        if completed.returncode != 0:
            raise SystemExit(f"torchrun failed with code {completed.returncode}")
        return

    device = run_llm._pick_device(0)
    run_llm._ensure_tokens(_make_args(args, {"method": "adam"}))
    runs, trajectories = run_grid(args, rank=0, world=1, device=device,
                                  process_group=None)
    write_artifacts(runs, trajectories, args.eps)


def _rebuild_argv(args) -> list[str]:
    out = [
        "--model", args.model, "--granularity", args.granularity,
        "--steps", str(args.steps), "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--gate-lr", str(args.gate_lr),
        "--dual-lr", str(args.dual_lr), "--dtype", args.dtype,
        "--eval-batches", str(args.eval_batches), "--seed", str(args.seed),
        "--eps", *[str(e) for e in args.eps],
        "--lambdas", *[str(l) for l in args.lambdas],
    ]
    if args.tokens:
        out += ["--tokens", args.tokens]
    if args.vocab_size is not None:
        out += ["--vocab-size", str(args.vocab_size)]
    if args.synthetic:
        out.append("--synthetic")
    if args.quick:
        out.append("--quick")
    return out


if __name__ == "__main__":
    main()
