"""
E3 — do the dual methods deliver a requested sparsity on a large LM, and at what cost?

Sweeps every constrained method in :mod:`paper.e3.run_llm` against three target densities,
over three seeds, with two baselines beside them: ``adam`` -- the same gated model trained
with the loss alone, so the perplexity column has a reference -- and, when it is switched
on in ``run_llm.METHODS``, the fixed-penalty formulation of arXiv:2208.04425 Eq. (1)-(2)
that their Fig. 1 compares against. Three numbers per run:

* ``achieved - eps`` — did the method land on the density that was asked for, with the
  spread over seeds beside it;
* ``val_ppl`` / ``test_ppl`` — what the model is worth there, at the median (test-time)
  gates, reported next to ``median density``, which is what that network actually
  realises (the constraint acts on the *expected* density, which differs). The shard is
  split train | val | test; ``tune.py`` scores val, so test is a number nothing has
  selected on. ``--eval-every`` also writes both as curves over training;
* ``tokens_per_s`` and the step-time breakdown — what the dual layer cost to get it.

The headline is the **final iterate**, not an average over a tail of steps. E2a averages,
because its constraint is a sample mean whose sign flickers with the minibatch; here the
constraint is a closed form in the gate parameters, with no data and no sampling noise in
it, so the last value *is* the achieved density. The *primal* iterate is still stochastic
and the trajectory still oscillates around the target, which is what the seed spread in
the table is there to show. The full trajectory is written anyway.

**Distributed by necessity, not as a subject.** At 0.5 B parameters the grid does not run
on one GPU. Every rank walks the same list of configurations at full world size and rank 0
writes; the model is dropped between configurations because the whole grid in one process
otherwise fragments the allocator.

**No predictions are registered** — see ``run_llm.py``'s docstring. ``--check`` is accepted
and exits 0.

Usage::

    python paper/e3/sweep.py --quick                 # CPU stub, seconds
    python paper/e3/sweep.py --quick --ranks 2       # same, 2 gloo ranks
    python paper/e3/sweep.py --model Qwen/Qwen2.5-0.5B --tokens .../tokens.bin --ranks 2
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
# The final-iterate density oscillates around the target by about the margin separating
# the methods, so one seed cannot rank them.
SEEDS = (0, 1, 2)
# No target and no duals: the gated model trained with the loss alone. Every constrained
# row's perplexity and throughput is read against this one.
REFERENCES = ("adam",)


def _configs(args):
    """The grid, as (label, overrides) pairs; the seed is one of the overrides."""
    out = []
    for seed in args.seeds:
        for method in REFERENCES:
            if method in run_llm.METHODS:
                out.append((method, {"method": method, "seed": seed}))
        for method in run_llm.CONSTRAINED:
            for eps in args.eps:
                out.append((f"{method}@{eps:g}",
                            {"method": method, "eps": eps, "seed": seed}))
        # The baseline is skipped when it is switched off in run_llm.METHODS, so the
        # grid follows that one list the way CONSTRAINED does.
        if "penalty" in run_llm.METHODS:
            for lam in args.lambdas:
                out.append((f"penalty@{lam:g}",
                            {"method": "penalty", "penalty": lam, "seed": seed}))
    return out


def _make_args(args, overrides):
    """A single-run namespace: run_llm's defaults, this driver's flags, the overrides."""
    one = run_llm.build_parser().parse_args([])
    for key in ("model", "tokens", "synthetic", "vocab_size", "granularity", "steps",
                "seq_len", "batch_size", "lr", "gate_lr", "dual_lr", "dtype",
                "quick",
                "eval_batches", "val_frac", "test_frac", "eval_every", "test_curve"):
        setattr(one, key, getattr(args, key))
    for key, value in overrides.items():
        setattr(one, key, value)
    return run_llm.finalize(one)


def run_grid(args, *, rank, world, device, process_group):
    runs, trajectories, medians, curves = [], [], [], []
    configs = _configs(args)
    for index, (label, overrides) in enumerate(configs, start=1):
        seed = overrides["seed"]
        if rank == 0:
            print(f"\n[{index}/{len(configs)}] {label}  seed={seed}", flush=True)
        result = run_llm.train(
            _make_args(args, overrides), rank=rank, world=world,
            device=device, process_group=process_group,
        )
        if rank == 0:
            summary = dict(result["summary"])
            summary["label"] = label
            summary["seed"] = seed
            summary["penalty"] = overrides.get("penalty", "")
            if summary["method"] == "penalty" or summary["method"] in REFERENCES:
                # Neither formulation has a target; leaving run_llm's default eps in the
                # row would read as one.
                summary["eps"] = ""
            runs.append(summary)
            for row in result["rows"]:
                trajectories.append({"label": label, "seed": seed, **row})
            for row in result["median_report"]:
                medians.append({"label": label, "seed": seed, **row})
            for row in result["eval_rows"]:
                curves.append({"label": label, "seed": seed, **row})
        # The next configuration builds a fresh model; without this the previous one's
        # activations and optimizer state stay resident until the allocator is pressured.
        del result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return runs, trajectories, medians, curves


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #


def _mean(values):
    return sum(values) / len(values)


def _spread(values):
    """Half the range across seeds. With three of them that says more than a standard
    deviation, and it is what the ``+/-`` column means."""
    return (max(values) - min(values)) / 2.0 if len(values) > 1 else 0.0


def _summaries(runs):
    """The headline table, aggregated over seeds, and the two baselines beside it."""
    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["eps"], run["penalty"]), []).append(run)

    constrained, penalty, reference = [], [], []
    for (method, eps, lam), rows in groups.items():
        density = [r["final_density_mean"] for r in rows]
        shared = {
            "seeds": len(rows),
            "achieved density": _mean(density),
            "achieved +/-": _spread(density),
            # The density the median (test-time) network realises, which is what the
            # perplexity beside it was measured on.
            "median density": _mean([r.get("final_median_density", float("nan"))
                                     for r in rows]),
            "val ppl": _mean([r.get("val_ppl", float("nan")) for r in rows]),
            "test ppl": _mean([r.get("test_ppl", float("nan")) for r in rows]),
            "tokens/s": _mean([r["tokens_per_s"] for r in rows]),
        }
        if method in REFERENCES:
            reference.append({"method": method, **shared})
        elif method == "penalty":
            penalty.append({
                "lambda": lam,
                **shared,
                **{f"distance to eps={e:g}": _mean(density) - e for e in EPS},
            })
        else:
            constrained.append({
                "method": method,
                "eps": eps,
                "seeds": len(rows),
                "achieved density": shared["achieved density"],
                "achieved +/-": shared["achieved +/-"],
                "achieved - eps": _mean(density) - eps,
                "median density": shared["median density"],
                "per-layer min": _mean([r["final_density_min"] for r in rows]),
                "per-layer max": _mean([r["final_density_max"] for r in rows]),
                "val ppl": shared["val ppl"],
                "test ppl": shared["test ppl"],
                "tokens/s": shared["tokens/s"],
                "ms/step": _mean([r["ms_per_step"] for r in rows]),
                "ms dual": _mean([r["ms_dual"] for r in rows]),
            })
    constrained.sort(key=lambda r: (r["eps"], r["method"]))
    penalty.sort(key=lambda r: r["lambda"])
    return constrained, penalty, reference


def _figures(runs, trajectories, eps_values, seeds):
    by_key = {}
    for row in trajectories:
        by_key.setdefault((row["label"], row["seed"]), []).append(row)
    shown = seeds[0]

    # One panel per target: every method's density trajectory against the line it was
    # asked to reach. One seed only -- the rest are in the CSV, and three overlaid traces
    # per method make the panel unreadable.
    fig, axes, plt = figure(1, len(eps_values), row_height=2.4)
    for ax, eps in zip(axes, eps_values):
        for (label, seed), rows in sorted(by_key.items()):
            if seed != shown or label.startswith("penalty"):
                continue
            if not label.endswith(f"@{eps:g}"):
                continue
            ax.plot([r["step"] for r in rows], [r["density_mean"] for r in rows],
                    lw=1.0, label=label.split("@")[0])
        ax.axhline(eps, color="k", ls="--", lw=0.8)
        ax.set_title(f"$\\varepsilon$ = {eps:g}")
        ax.set_xlabel("step")
    axes[0].set_ylabel(f"expected density (seed {shown})")
    axes[-1].legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "e3_sweep_trajectories", EXPERIMENT)
    plt.close(fig)

    # Achieved against requested, with the penalty sweep as a horizontal spread: it has
    # no target to be plotted against, which is the point being made.
    fig, axes, plt = figure(1, 1, row_height=2.6)
    ax = axes[0]
    methods = sorted({r["method"] for r in runs
                      if r["method"] != "penalty" and r["method"] not in REFERENCES})
    for method in methods:
        by_eps = {}
        for run in runs:
            if run["method"] == method:
                by_eps.setdefault(run["eps"], []).append(run["final_density_mean"])
        targets = sorted(by_eps)
        # Bars are the seed spread, not a standard error: three runs, same settings.
        ax.errorbar(targets, [_mean(by_eps[t]) for t in targets],
                    yerr=[_spread(by_eps[t]) for t in targets],
                    marker="o", ms=3, lw=1.0, capsize=2, label=method)
    penalties = [r["final_density_mean"] for r in runs if r["method"] == "penalty"]
    for value in penalties:
        ax.axhline(value, color="grey", lw=0.6, alpha=0.6)
    for method in REFERENCES:
        for run in runs:
            if run["method"] == method:
                ax.axhline(run["final_density_mean"], color="C7", ls=":", lw=0.8)
                break
    lo, hi = min(eps_values), max(eps_values)
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8, label="target")
    ax.set_xlabel("requested density $\\varepsilon$")
    ax.set_ylabel("achieved density")
    if penalties:
        ax.set_title("grey: fixed-penalty runs (no target to request)")
    ax.legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "e3_sweep_achieved", EXPERIMENT)
    plt.close(fig)


def _curve_figure(curves, eps_values, seeds):
    """Validation and test perplexity over training, one panel per target.

    Step-indexed, like the density trajectory it is meant to be read beside: at 5000
    steps the sweep covers about two epochs of the shard, so this is also where a
    constraint that costs the model something shows up.
    """
    by_key = {}
    for row in curves:
        if "val_ppl" in row:
            by_key.setdefault((row["label"], row["seed"]), []).append(row)
    shown = seeds[0]
    fig, axes, plt = figure(1, len(eps_values), row_height=2.4)
    for ax, eps in zip(axes, eps_values):
        for (label, seed), rows in sorted(by_key.items()):
            if seed != shown or not label.endswith(f"@{eps:g}"):
                continue
            rows = sorted(rows, key=lambda r: r["step"])
            line, = ax.plot([r["step"] for r in rows], [r["val_ppl"] for r in rows],
                            lw=1.0, label=label.split("@")[0])
            held = [r for r in rows if "test_ppl" in r]
            if held:
                ax.plot([r["step"] for r in held], [r["test_ppl"] for r in held],
                        lw=0.8, ls=":", color=line.get_color())
        for (label, seed), rows in sorted(by_key.items()):
            # The unconstrained reference, as a flat comparison on every panel.
            if seed == shown and label in REFERENCES:
                rows = sorted(rows, key=lambda r: r["step"])
                ax.plot([r["step"] for r in rows], [r["val_ppl"] for r in rows],
                        lw=1.0, color="k", ls="--", label=label)
        ax.set_title(f"$\\varepsilon$ = {eps:g}")
        ax.set_xlabel("step")
    axes[0].set_ylabel(f"perplexity, median gates (seed {shown})")
    axes[-1].legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, "e3_sweep_curves", EXPERIMENT)
    plt.close(fig)


def write_artifacts(runs, trajectories, medians, curves, eps_values, seeds):
    write_csv(trajectories, "e3_sweep_trajectories", EXPERIMENT)
    write_csv(runs, "e3_sweep_runs", EXPERIMENT)
    if medians:
        write_csv(medians, "e3_sweep_median_density", EXPERIMENT)
    if curves:
        write_csv(curves, "e3_sweep_curves", EXPERIMENT)
    constrained, penalty, reference = _summaries(runs)
    if constrained:
        write_table(constrained, "e3_sweep", EXPERIMENT,
                    title="E3: achieved density (mean over seeds, +/- half the range), "
                          "the density the median network realises, held-out perplexity "
                          "there, and throughput (final iterate)")
    if reference:
        write_table(reference, "e3_reference", EXPERIMENT,
                    title="E3: the gated model with no constraint, as the perplexity "
                          "and throughput reference")
    if penalty:
        write_table(penalty, "e3_penalty", EXPERIMENT,
                    title="E3: the fixed-penalty baseline, one coefficient per run")
    _figures(runs, trajectories, eps_values, seeds)
    if curves:
        _curve_figure(curves, eps_values, seeds)
    print()
    for row in reference:
        print(f"  {row['method']:<17} (no target)  "
              f"density {row['achieved density']:.4f}             "
              f"val {row['val ppl']:.2f}  test {row['test ppl']:.2f}  "
              f"{row['tokens/s']:.0f} tok/s")
    for row in constrained:
        print(f"  {row['method']:<17} eps={row['eps']:.2f}  "
              f"achieved {row['achieved density']:.4f} "
              f"(+/-{row['achieved +/-']:.4f}, {row['achieved - eps']:+.4f})  "
              f"val {row['val ppl']:.2f}  test {row['test ppl']:.2f}  "
              f"{row['tokens/s']:.0f} tok/s")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub")
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--granularity", default="block")
    parser.add_argument("--eps", type=float, nargs="+", default=list(EPS))
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8, help="per rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--dual-lr", type=float, default=1e-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="steps between val/test curve points; 0 for endpoints only")
    parser.add_argument("--test-curve", action="store_true", default=True)
    parser.add_argument("--no-test-curve", dest="test_curve", action="store_false")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="one full grid per seed; the table reports the spread")
    parser.add_argument("--ranks", type=int, default=None,
                        help="ranks to launch; default is every visible GPU")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="accepted for parity; E3 registers no predictions")
    args = parser.parse_args(argv)

    if args.quick:
        args.eps = args.eps[:2]
        args.lambdas = args.lambdas[:2]
        args.seeds = args.seeds[:1]
        args.steps = 6

    rank, world, local_rank = run_llm._dist_env()

    if rank >= 0:  # a worker under torchrun
        device = run_llm._pick_device(local_rank, world)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        try:
            runs, trajectories, medians, curves = run_grid(
                args, rank=rank, world=world, device=device,
                process_group=dist.group.WORLD,
            )
            if rank == 0:
                write_artifacts(runs, trajectories, medians, curves,
                                args.eps, args.seeds)
        finally:
            dist.barrier()
            dist.destroy_process_group()
        return

    ranks = args.ranks if args.ranks is not None else max(1, torch.cuda.device_count())
    if ranks > 1:
        # The shard must exist before the ranks race for it; run_llm writes it only in
        # its own single-process path.
        run_llm._ensure_tokens(
            _make_args(args, {"method": "adam", "seed": args.seeds[0]}))
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
    run_llm._ensure_tokens(
        _make_args(args, {"method": "adam", "seed": args.seeds[0]}))
    runs, trajectories, medians, curves = run_grid(args, rank=0, world=1, device=device,
                                                   process_group=None)
    write_artifacts(runs, trajectories, medians, curves, args.eps, args.seeds)


def _rebuild_argv(args) -> list[str]:
    out = [
        "--model", args.model, "--granularity", args.granularity,
        "--steps", str(args.steps), "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--gate-lr", str(args.gate_lr),
        "--dual-lr", str(args.dual_lr), "--dtype", args.dtype,
        "--eval-batches", str(args.eval_batches),
        "--val-frac", str(args.val_frac), "--test-frac", str(args.test_frac),
        "--eval-every", str(args.eval_every),
        "--seeds", *[str(s) for s in args.seeds],
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
    if not args.test_curve:
        out.append("--no-test-curve")
    return out


if __name__ == "__main__":
    main()
