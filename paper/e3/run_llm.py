"""
E3 — sparsity-constrained LM fine-tuning under data parallelism.

Implements Eq. (3) of Gallego-Posada, Ramirez, Erraqabi, Bengio & Lacoste-Julien,
*Controlled Sparsity via Constrained Optimization* (NeurIPS 2022, arXiv:2208.04425):
structured hard-concrete L0 gates on a decoder LM, with per-block or model-wide
constraints on the expected parameter density. The gates and constraints live in
:mod:`paper.problems.sparse_lm`; this script is the training loop, the instrumentation
and the artifact writing.

**This file is one run.** The experiment is the two drivers on top of it:
``sweep.py`` (methods x target density, plus the fixed-penalty baseline) and ``scaling.py``
(tokens/s over the 1->2->4 GPU ladder). Both build their configurations from
:func:`build_parser`, so a default added here reaches them.

**What E3 asks.** On a large multi-GPU LM job: do the dual methods deliver a requested
density, what is the model worth at that density (perplexity at the median gates, on
held-out blocks), and what does the dual layer cost in throughput. It does **not** validate
the data-parallel constraint reduction — the density constraint is a closed form in the gate
parameters with no data in it, so every rank computes the same value and ``all_reduce(c,
AVG)`` cannot be caught being wrong here. That claim belongs to ``paper/e2/b_parallel.py``,
whose constraints are sample averages.

**This script registers no predictions**, and neither do the drivers. The other paper
experiments state falsifiable expectations up front and gate on them with ``--check``; E3 is
deliberately descriptive, so ``--check`` is accepted for interface parity and exits 0.
Assertions about the gate mathematics live in ``tests/test_llm_gates.py``, where they
belong.

**SSG is deliberately not wired here.** It is implemented in the package but set aside for
every paper experiment, and its switching step takes a different code path (it steps the
model on ``max_i c_i`` rather than on a surrogate), so adding it would need its own branch
below as well as revisiting that scope decision.

Usage::

    # CPU smoke, no downloads and no `transformers` needed
    python paper/e3/run_llm.py --model stub --synthetic --quick
    python paper/e3/run_llm.py --model stub --synthetic --quick --ranks 2

    # the real thing, under torchrun (see sbatch_e3.sh)
    torchrun --standalone --nproc_per_node=4 paper/e3/run_llm.py \
        --model Qwen/Qwen2.5-0.5B --tokens paper/results/e3/tokens.bin \
        --method alm_gda_restart --granularity layer --eps 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from humancompatible.train.dual_optim import ALM, PBM, iALM, nuPI
from paper._harness import (
    RESULTS,
    figure,
    save_figure,
    set_seed,
    write_csv,
    write_table,
)
from paper.problems import sparse_lm, tokens as tokens_mod

EXPERIMENT = "e3"


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

# Each builder takes (m, device, process_group) and returns a DualOptimizer, or None for
# the two references that have no duals. Hyperparameter values are transcribed rather than
# read from `benchmark/new_bench/conf/algorithm/*.yaml`, so this tree carries no Hydra
# dependency -- the same choice `paper/e0/d_distributed.py` makes.
#
# `alm_gda` and `alm_gda_restart` are the reference method: ALM(penalty=0, is_ineq=True) is
# the projected gradient descent-ascent of arXiv:2208.04425 Eq. (5), and restart=True is
# that paper's dual restart, Eq. (6).
METHODS = {
    # references, no dual optimizer
    "adam": None,
    "penalty": None,
    # the reference method and its variants
    "alm_gda": lambda m, device, pg, lr: ALM(
        m=m, lr=lr, penalty=0.0, init_duals=0.0, is_ineq=True,
        device=device, process_group=pg,
    ),
    "alm_gda_restart": lambda m, device, pg, lr: ALM(
        m=m, lr=lr, penalty=0.0, init_duals=0.0, is_ineq=True, restart=True,
        device=device, process_group=pg,
    ),
    "alm_quad": lambda m, device, pg, lr: ALM(
        m=m, lr=lr, penalty=1.0, init_duals=0.0, is_ineq=True,
        device=device, process_group=pg,
    ),
    "ialm": lambda m, device, pg, lr: iALM(
        m=m, beta=lr, sigma=1.0, gamma=1.0, init_duals=0.0, is_ineq=True,
        device=device, process_group=pg,
    ),
    "nupi": lambda m, device, pg, lr: nuPI(
        m=m, ki=lr, kp=lr, nu=0.0, penalty=0.0, init_duals=0.0, is_ineq=True,
        device=device, process_group=pg,
    ),
    # Annealing is switched off explicitly: it defaults on and then makes `epoch_length`
    # mandatory, so PBM(m=...) alone raises. Recorded as a blocker in the plan.
    "pbm": lambda m, device, pg, lr: PBM(
        m=m, gamma=0.5, penalty_mult=lr, delta=1.0, penalty_update="dimin_adapt",
        gamma_annealing=False, penalty_annealing=False,
        device=device, process_group=pg,
    ),
}

CONSTRAINED = tuple(name for name, build in METHODS.items() if build is not None)


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #


class StepTimer:
    """Phase timings for one step, honest on GPU as well as CPU.

    On CUDA every mark is a ``torch.cuda.Event`` and the elapsed times are read after a
    single ``synchronize()`` at the end of the step. Nothing in this repository did that
    before: the existing per-epoch ``time`` columns in ``benchmark/`` use bare
    ``perf_counter`` around asynchronous launches, so they measure launch time rather than
    compute, and would report the dual layer as free for the wrong reason.
    """

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.marks: list[tuple[str, object]] = []

    def mark(self, name: str) -> None:
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks.append((name, event))
        else:
            self.marks.append((name, time.perf_counter()))

    def finish(self) -> dict:
        if self.cuda:
            torch.cuda.synchronize()
        out = {}
        for (_, before), (name, after) in zip(self.marks, self.marks[1:]):
            if self.cuda:
                out[f"ms_{name}"] = before.elapsed_time(after)
            else:
                out[f"ms_{name}"] = (after - before) * 1e3
        out["ms_total"] = sum(v for k, v in out.items() if k.startswith("ms_"))
        return out


def comm_accounting(model: nn.Module, m: int, world: int, dtype_bytes: int = 4) -> dict:
    """Bytes the two collectives move per step.

    Analytic, and labelled as such. The gradient payload is the trainable-parameter
    footprint, which DDP all-reduces once per step in buckets; the constraint payload is
    the ``m``-vector. The ring-corrected figures are the payload times
    ``2(world-1)/world``, which is what a ring all-reduce actually puts on the wire.
    """
    grad = sparse_lm.trainable_bytes(model)
    dual = m * dtype_bytes
    factor = 0.0 if world < 2 else 2.0 * (world - 1) / world
    return {
        "grad_payload_bytes": grad,
        "dual_payload_bytes": dual,
        "grad_ring_bytes": grad * factor,
        "dual_ring_bytes": dual * factor,
        "dual_over_grad": dual / grad if grad else float("nan"),
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def build_model(name: str, *, device, dtype, seed: int, vocab_size: int = None):
    """The stand-in model, or a HuggingFace causal LM.

    ``transformers`` is imported lazily and only for a real model name, so the whole smoke
    path runs without it installed — which is the situation both on this machine and on the
    cluster, where no Transformers module exists.
    """
    if name == "stub":
        from paper.problems.tiny_lm import tiny_causal_lm

        kwargs = {} if vocab_size is None else {"vocab_size": vocab_size}
        model = tiny_causal_lm(seed=seed, **kwargs)
        return model.to(device=device, dtype=dtype)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            f"loading {name!r} needs `transformers`, which is not installed. Use "
            f"`--model stub` for the smoke path, or install transformers on the cluster "
            f"(no EasyBuild module provides it)."
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.config.use_cache = False  # incompatible with training, and wastes memory
    return model.to(device)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def evaluate(model, shard, first_block, n_blocks, *, batch_size, device, rank, world):
    """Mean cross-entropy over held-out blocks, at whatever gates are installed.

    Blocks are handed out contiguously per rank and truncated to an equal count, so the
    ``ReduceOp.AVG`` over ranks is the mean over all of them. Returns ``None`` when the
    shard is too small to give every rank a batch, which is the ``--quick`` case.
    """
    per_rank = n_blocks // world
    if per_rank < batch_size:
        return None
    total, batches = 0.0, per_rank // batch_size
    with torch.no_grad():
        for b in range(batches):
            start = first_block + rank * per_rank + b * batch_size
            ids = shard.batch(range(start, start + batch_size), device=device)
            total += float(model(input_ids=ids, labels=ids).loss)
    mean = torch.tensor(total / batches, device=device)
    if world > 1:
        dist.all_reduce(mean, op=dist.ReduceOp.AVG)
    return float(mean)


def train(args, *, rank: int, world: int, device, process_group) -> dict:
    """Run ``args.steps`` steps; return per-step rows plus a summary."""
    set_seed(args.seed)
    dtype = torch.float32 if device.type == "cpu" else getattr(torch, args.dtype)

    shard = tokens_mod.TokenShard(args.tokens, args.seq_len)
    model = build_model(
        args.model, device=device, dtype=dtype, seed=args.seed,
        vocab_size=args.vocab_size,
    )

    gates = None
    if not args.no_gates:
        gates = sparse_lm.attach_gates(
            model,
            gate_mlp=not args.no_mlp_gates,
            gate_heads=not args.no_head_gates,
            init_open=args.init_open,
            seed=args.seed,
            device=device,
        )
    facts = sparse_lm.describe(model, gates)

    m = 0 if gates is None else gates.m(args.granularity)
    names = [] if gates is None else gates.constraint_names(args.granularity)

    # Gates must be attached before the DDP wrap so their parameters are registered and
    # synchronised like any other.
    wrapped = model
    if world > 1:
        wrapped = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    # Gate parameters get their own, much larger step. Adam moves a parameter by about
    # its learning rate per step regardless of gradient scale, and carrying log_alpha
    # from a 95%-open initialisation to a 30%-density solution is a move of ~3.8 -- at
    # the model's 1e-4 that is 38k steps, so a shared step size would make this
    # experiment a measurement of the learning rate rather than of the methods.
    # Separate optimizers for weights and gates is also what arXiv:2208.04425 does.
    gate_ids = set() if gates is None else {id(p) for p in gates.gate_parameters()}
    groups = [
        {"params": [p for p in wrapped.parameters() if id(p) not in gate_ids],
         "lr": args.lr},
    ]
    if gate_ids:
        groups.append({"params": [p for p in wrapped.parameters() if id(p) in gate_ids],
                       "lr": args.gate_lr})
    primal = torch.optim.AdamW(groups, lr=args.lr)
    dual = None
    build = METHODS[args.method]
    if build is not None:
        if gates is None:
            raise SystemExit(f"--method {args.method} needs gates; drop --no-gates")
        dual = build(m, device, process_group, args.dual_lr)

    # A generator seeded identically on every rank makes the objective's gate noise
    # shared, which is what makes a `world x B` run comparable to `1 x (world*B)`. Seeding
    # per rank instead averages over `world` gate draws -- a different estimator, and the
    # reason this is a flag rather than a constant.
    gate_gen = None
    if gates is not None:
        gate_gen = torch.Generator(device=device)
        gate_gen.manual_seed(args.seed if args.sync_gates else args.seed + 9973 * rank)

    # The last blocks of the shard are held out. Training never sees them, so the
    # perplexity reported at the median gates is a number about the model rather than
    # about how much of the shard it memorised.
    held_out = min(args.eval_batches * args.batch_size * world, len(shard) // 4)
    n_train = len(shard) - held_out
    sampler = tokens_mod.BlockSampler(
        n_train, args.batch_size, rank=rank, world=world, seed=args.seed
    )
    batches = sampler.stream(args.steps)

    comm = comm_accounting(model, m, world)
    rows = []
    for step, indices in enumerate(batches):
        timer = StepTimer(device)
        timer.mark("start")

        ids = shard.batch(indices, device=device)
        if gates is not None:
            # Fresh gate draw per step. This must precede the forward pass: the sample is
            # what carries gradients to log_alpha, and reusing one across two backward
            # passes raises from autograd.
            gates.resample(gate_gen)

        loss = wrapped(input_ids=ids, labels=ids).loss
        timer.mark("forward")

        constraints = None
        if gates is not None:
            constraints = gates.constraints(args.eps, args.granularity)
        if dual is not None:
            objective = dual.forward_update(loss, constraints)
        elif args.method == "penalty":
            # Their penalized formulation, Eq. (1)-(2): a fixed multiple of the expected
            # L0 added to the loss. The sweep over `--penalty` is the comparison their
            # Fig. 1 makes against the constrained approach.
            objective = loss + args.penalty * gates.densities(args.granularity).mean()
        else:
            objective = loss
        timer.mark("dual")

        objective.backward()
        timer.mark("backward")

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.grad_clip)
        primal.step()
        primal.zero_grad(set_to_none=True)
        timer.mark("opt")

        row = {
            "step": step,
            "loss": float(loss.detach()),
            "tokens": args.batch_size * args.seq_len * world,
            **timer.finish(),
        }
        if gates is not None:
            with torch.no_grad():
                density = gates.densities(args.granularity)
            row["density_mean"] = float(density.mean())
            row["density_min"] = float(density.min())
            row["density_max"] = float(density.max())
            row["max_violation"] = float((density - args.eps).max())
        if dual is not None:
            with torch.no_grad():
                duals = dual.duals.detach()
            row["dual_mean"] = float(duals.mean())
            row["dual_max"] = float(duals.max())
            if world > 1:
                # Measured, not predicted: how far the ranks' duals are from rank 0's.
                gathered = [torch.zeros_like(duals) for _ in range(world)]
                dist.all_gather(gathered, duals)
                row["dual_spread_across_ranks"] = float(
                    max((g - gathered[0]).abs().max() for g in gathered)
                )
        rows.append(row)
        if rank == 0 and (step % max(1, args.steps // 10) == 0 or step == args.steps - 1):
            note = f"  density={row.get('density_mean', float('nan')):.4f}" if gates else ""
            print(f"  step {step:>5d}  loss={row['loss']:.4f}"
                  f"  {row['ms_total']:.1f} ms{note}", flush=True)

    warmup = min(args.warmup, max(0, len(rows) - 1))
    steady = rows[warmup:] or rows
    seconds = sum(r["ms_total"] for r in steady) / 1e3
    summary = {
        "method": args.method,
        "model": args.model,
        "world": world,
        "granularity": args.granularity if gates else "none",
        "m": m,
        "eps": args.eps,
        "per_rank_batch": args.batch_size,
        "seq_len": args.seq_len,
        "steps": len(rows),
        "warmup_steps_dropped": warmup,
        "tokens_per_s": sum(r["tokens"] for r in steady) / seconds if seconds else 0.0,
        "ms_per_step": sum(r["ms_total"] for r in steady) / len(steady),
        "final_loss": rows[-1]["loss"],
        **{k: sum(r[k] for r in steady) / len(steady)
           for k in ("ms_forward", "ms_backward", "ms_dual", "ms_opt")},
        **facts,
        **comm,
    }
    # The test-time model is the median gate (their Appendix A.1), not a fresh sample:
    # what a user ships is one deterministic network, and its perplexity is what says
    # whether the density the constraint delivered was worth having.
    if gates is not None:
        gates.use_median()
    eval_loss = evaluate(
        model, shard, n_train, held_out, batch_size=args.batch_size,
        device=device, rank=rank, world=world,
    )
    if eval_loss is not None:
        summary["eval_loss"] = eval_loss
        summary["eval_ppl"] = float(np.exp(eval_loss))
        summary["eval_blocks"] = held_out

    if gates is not None:
        summary["final_density_mean"] = rows[-1]["density_mean"]
        summary["final_density_min"] = rows[-1]["density_min"]
        summary["final_density_max"] = rows[-1]["density_max"]
        summary["final_max_violation"] = rows[-1]["max_violation"]
    if device.type == "cuda":
        summary["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)

    report = [] if gates is None else gates.median_report()
    return {"rows": rows, "summary": summary, "median_report": report, "names": names}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def _dist_env() -> tuple[int, int, int]:
    """``(rank, world, local_rank)`` from torchrun's environment; rank -1 if absent."""
    return (
        int(os.environ.get("RANK", -1)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _ensure_tokens(args) -> None:
    """Write a synthetic shard if asked and none exists, so the smoke path is one command."""
    path = Path(args.tokens)
    if path.exists() or not args.synthetic:
        return
    vocab = args.vocab_size or 256
    n = max(args.steps + 8, 64) * args.batch_size * args.seq_len * 4
    written = tokens_mod.write_synthetic_shard(path, n, vocab, seed=args.seed)
    print(f"wrote a synthetic shard: {written} tokens -> {path}")


def build_parser() -> argparse.ArgumentParser:
    """The single-run flags. ``sweep.py`` and ``scaling.py`` build their configurations
    from ``build_parser().parse_args([])`` rather than assembling an argv, so a default
    added here reaches every driver."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub",
                        help="'stub' for the transformers-free stand-in, else a HF id")
    parser.add_argument("--tokens", default=None, help="path to a uint16 token shard")
    parser.add_argument("--synthetic", action="store_true",
                        help="write a random-token shard if --tokens does not exist")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--method", default="alm_gda_restart", choices=sorted(METHODS))
    parser.add_argument("--granularity", default="layer",
                        choices=sparse_lm.GRANULARITIES)
    parser.add_argument("--eps", type=float, default=0.5, help="target density")
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="fixed penalty coefficient, for --method penalty")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5,
                        help="steps dropped before the throughput average")
    parser.add_argument("--eval-batches", type=int, default=16,
                        help="held-out batches per rank for the median-gate perplexity")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4, help="per rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2,
                        help="step size for the gate parameters; see train()")
    parser.add_argument("--dual-lr", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16",
                        help="model dtype on GPU; CPU always runs float32")
    parser.add_argument("--init-open", type=float, default=0.95)
    parser.add_argument("--no-gates", action="store_true",
                        help="the plain LM: no hooks, no constraints")
    parser.add_argument("--no-mlp-gates", action="store_true")
    parser.add_argument("--no-head-gates", action="store_true")
    parser.add_argument("--sync-gates", action="store_true", default=True)
    parser.add_argument("--no-sync-gates", dest="sync_gates", action="store_false",
                        help="draw independent gate samples per rank")
    parser.add_argument("--no-process-group", action="store_true",
                        help="skip the constraint all-reduce, to measure its redundancy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1,
                        help="re-exec under torchrun with this many ranks")
    parser.add_argument("--tag", default=None, help="suffix for the artifact names")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="accepted for parity; E3 registers no predictions")
    return parser


def finalize(args):
    """Apply ``--quick``, fill in the shard path, reject impossible combinations.

    Shared with the drivers, which override fields on a defaults namespace and would
    otherwise each have to remember what ``--quick`` means.
    """
    if args.quick:
        args.steps = min(args.steps, 6)
        args.seq_len = min(args.seq_len, 32)
        args.batch_size = min(args.batch_size, 2)
        args.warmup = 1
        args.eval_batches = min(args.eval_batches, 2)
        args.synthetic = True
    if args.tokens is None:
        args.tokens = str(RESULTS / EXPERIMENT / ("tokens_quick.bin" if args.quick
                                                  else "tokens.bin"))
    if args.method == "penalty" and args.no_gates:
        raise SystemExit("--method penalty needs gates")
    return args


def main(argv=None) -> None:
    args = finalize(build_parser().parse_args(argv))

    rank, world, local_rank = _dist_env()

    # Already under torchrun: be a worker.
    if rank >= 0:
        device = _pick_device(local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        group = None if args.no_process_group else dist.group.WORLD
        try:
            result = train(args, rank=rank, world=world, device=device,
                           process_group=group)
            if rank == 0:
                _write(args, result, world)
        finally:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Not under torchrun, but more than one rank was asked for: re-exec, as
    # paper/e0/d_distributed.py does, so this file stays the single entry point.
    if args.ranks > 1:
        _ensure_tokens(args)
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={args.ranks}",
            str(Path(__file__).resolve()),
        ] + _rebuild_argv(args)
        print(f"launching {args.ranks} ranks ...", flush=True)
        completed = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1"))
        if completed.returncode != 0:
            raise SystemExit(f"torchrun failed with code {completed.returncode}")
        return

    # Plain single process.
    _ensure_tokens(args)
    device = _pick_device(0)
    result = train(args, rank=0, world=1, device=device, process_group=None)
    _write(args, result, 1)


def _rebuild_argv(args) -> list[str]:
    """Reconstruct the flags for the re-exec, minus --ranks."""
    out = [
        "--model", args.model, "--tokens", args.tokens, "--method", args.method,
        "--granularity", args.granularity, "--eps", str(args.eps),
        "--penalty", str(args.penalty), "--steps", str(args.steps),
        "--warmup", str(args.warmup), "--eval-batches", str(args.eval_batches),
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--gate-lr", str(args.gate_lr),
        "--dual-lr", str(args.dual_lr), "--grad-clip", str(args.grad_clip),
        "--dtype", args.dtype, "--init-open", str(args.init_open),
        "--seed", str(args.seed),
    ]
    if args.vocab_size is not None:
        out += ["--vocab-size", str(args.vocab_size)]
    for flag, on in (
        ("--no-gates", args.no_gates),
        ("--no-mlp-gates", args.no_mlp_gates),
        ("--no-head-gates", args.no_head_gates),
        ("--no-process-group", args.no_process_group),
        ("--synthetic", args.synthetic),
        ("--quick", args.quick),
    ):
        if on:
            out.append(flag)
    if not args.sync_gates:
        out.append("--no-sync-gates")
    if args.tag:
        out += ["--tag", args.tag]
    return out


def _slug(args, world: int) -> str:
    """Artifact name. ``world`` comes from the live process group, never from
    ``args.ranks``: the re-exec strips ``--ranks``, so a worker always reads it as 1 and
    every multi-rank run would overwrite the single-process artifacts of the same method."""
    base = f"{args.method}_{args.granularity}_eps{args.eps:g}_w{world}"
    if args.no_gates:
        base += "_nogates"
    if args.no_process_group and world > 1:
        base += "_nopg"
    return f"{base}_{args.tag}" if args.tag else base


def _write(args, result: dict, world: int) -> None:
    slug = _slug(args, world)
    write_csv(result["rows"], f"e3_steps_{slug}", EXPERIMENT)
    write_table([result["summary"]], f"e3_summary_{slug}", EXPERIMENT,
                title=f"E3: {args.method} on {args.model}, "
                      f"{args.granularity} granularity, eps={args.eps:g}")
    if result["median_report"]:
        write_table(result["median_report"], f"e3_density_{slug}", EXPERIMENT,
                    title="E3: per-group density at the median gates")
    _make_figure(result, slug)
    print(json.dumps(result["summary"], indent=2, default=str))
    if args.check:
        print("\nE3 registers no predictions; --check has nothing to gate on.")


def _make_figure(result: dict, slug: str) -> None:
    rows = result["rows"]
    if len(rows) < 2 or "density_mean" not in rows[0]:
        return
    fig, axes, plt = figure(1, 2, row_height=2.2)
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, [r["loss"] for r in rows], lw=1.0)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss")
    axes[1].plot(steps, [r["density_mean"] for r in rows], lw=1.0, label="mean")
    axes[1].fill_between(steps, [r["density_min"] for r in rows],
                         [r["density_max"] for r in rows], alpha=0.25,
                         label="min-max over constraints")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("expected density")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, f"e3_{slug}", EXPERIMENT)
    plt.close(fig)


if __name__ == "__main__":
    main()
