"""
E3 — sparsity-constrained LM fine-tuning under data parallelism.

Implements Eq. (3) of Gallego-Posada et al., *Controlled Sparsity via Constrained
Optimization* (NeurIPS 2022, arXiv:2208.04425): structured hard-concrete L0 gates on a
decoder LM, with per-block or model-wide constraints on the expected parameter density.
Gates and constraints live in :mod:`paper.problems.sparsity.sparse_lm`; this file is one
training run. ``sweep.py`` and ``scaling.py`` build their configurations from
:func:`build_parser`, so a default added here reaches them.

The density constraint is a closed form in the gate parameters, identical on every rank, so
this experiment does not test the data-parallel constraint reduction (that is E0d's job).

Usage::

    # CPU smoke, no downloads and no `transformers` needed
    python paper/e3/run_llm.py --model stub --synthetic --quick
    python paper/e3/run_llm.py --model stub --synthetic --quick --ranks 2
    python paper/e3/run_llm.py --model stub --synthetic --quick --method penalty --penalty 0.3

    # the real thing, under torchrun (see sbatch_e3.sh)
    torchrun --standalone --nproc_per_node=4 paper/e3/run_llm.py \
        --model Qwen/Qwen2.5-0.5B --tokens paper/results/e3/tokens.bin \
        --method alm_gda_restart --granularity block --eps 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from humancompatible.train.dual_optim import ALM, PBM, iALM, nuPI
from paper._harness import RESULTS, figure, save_figure, set_seed, write_csv, write_table
from paper.problems import tokens as tokens_mod
from paper.problems.sparsity import sparse_lm, sparsity_gates

EXPERIMENT = "e3"


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #

# `(class, kwargs)`, or None for a method with no dual optimizer. Each method's dual step
# size is its own named hyperparameter on purpose (ALM `lr`, iALM `beta`, nuPI `ki`/`kp`,
# PBM `penalty_mult`); override with `--dual-opt KEY=VALUE`.
METHODS = {
    # no duals
    "adam": None,
    # fixed penalty, Eq. (1)-(2): loss + --penalty * mean expected density
    "penalty": None,
    # ALM(penalty=0, is_ineq) is the projected GDA of Eq. (5); restart=True is Eq. (6)
    "alm_gda": (ALM, dict(
        lr=1e-2, penalty=0.0, init_duals=0.0, is_ineq=True,
    )),
    "alm_gda_restart": (ALM, dict(
        lr=1e-2, penalty=0.0, init_duals=0.0, is_ineq=True, restart=True,
    )),
    "alm_hpr": (ALM, dict(
        lr=1e-2, penalty=1.0, init_duals=0.0, is_ineq=True,
        augmentation="hpr", restart=True,
    )),
    "ialm": (iALM, dict(
        beta=1e-2, sigma=1.0, gamma=1.0, init_duals=0.0, is_ineq=True,
    )),
    "nupi": (nuPI, dict(
        ki=1e-2, kp=1e-2, nu=0.0, init_duals=0.0, is_ineq=True,
    )),
    # annealing defaults on and then requires `epoch_length`, so switch it off explicitly
    "pbm": (PBM, dict(
        gamma=0.5, penalty_mult=1e-2, delta=1.0, penalty_update="dimin_adapt",
        gamma_annealing=False, penalty_annealing=False,
    )),
}

CONSTRAINED = tuple(name for name, spec in METHODS.items() if spec is not None)


def _coerce(text: str):
    """argv is strings; constructors want floats, ints, bools and the odd name."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_dual_opts(items) -> dict:
    """``["kp=0.5", "nu=0.1"]`` -> ``{"kp": 0.5, "nu": 0.1}``."""
    out = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--dual-opt expects KEY=VALUE; got {item!r}")
        out[key.strip()] = _coerce(value)
    return out


def build_dual(method: str, m: int, *, device, process_group, overrides: dict = None):
    """This method's dual optimizer, with ``overrides`` on top of its defaults.

    None of the constructors take ``**kwargs``, so a typo in ``--dual-opt`` raises
    ``TypeError`` instead of silently running the default.
    """
    spec = METHODS[method]
    if spec is None:
        return None
    cls, defaults = spec
    kwargs = {**defaults, **(overrides or {})}  # a copy: sweep.py reuses METHODS in-process
    return cls(m=m, device=device, process_group=process_group, **kwargs)


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #


class StepTimer:
    """Phase timings for one step. On CUDA, events read after one ``synchronize()``, so
    the numbers are compute time rather than kernel-launch time."""

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
        out["ms_total"] = sum(out.values())
        return out


def comm_accounting(model: nn.Module, m: int, world: int, dtype_bytes: int = 4) -> dict:
    """Analytic bytes per step: DDP's gradient all-reduce vs the m-vector constraint one.
    Ring figures are payload times ``2(world-1)/world``."""
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


def build_model(name: str, *, device, seed: int, vocab_size: int = None):
    """The stand-in model or a HuggingFace causal LM, always in float32 (``--dtype`` is
    only the autocast dtype: bf16 master weights would drop most AdamW updates at 1e-4)."""
    if name == "stub":
        from paper.problems.tiny_lm import tiny_causal_lm

        kwargs = {} if vocab_size is None else {"vocab_size": vocab_size}
        model = tiny_causal_lm(seed=seed, **kwargs)
        return model.to(device=device, dtype=torch.float32)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            f"loading {name!r} needs `transformers`; use `--model stub` for the smoke path"
        ) from exc
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.config.use_cache = False  # incompatible with training, and wastes memory
    return model.to(device)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def evaluate(model, shard, first_block, n_blocks, *, batch_size, device, rank, world,
             amp_dtype=None, max_batches=None):
    """Mean cross-entropy over a split, at whatever gates are installed.

    Blocks are split contiguously and equally per rank, so the rank average is the mean
    over all of them. ``max_batches`` (per rank, 0/None = all) always takes the split's
    first blocks. Returns None when the split cannot give every rank a batch.
    """
    per_rank = n_blocks // world
    if per_rank < batch_size:
        return None
    batches = per_rank // batch_size
    if max_batches:
        batches = min(batches, max_batches)
    total = 0.0
    with torch.no_grad():
        for b in range(batches):
            start = first_block + rank * per_rank + b * batch_size
            ids = shard.batch(range(start, start + batch_size), device=device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                total += float(model(input_ids=ids, labels=ids).loss)
    mean = torch.tensor(total / batches, device=device)
    if world > 1:
        dist.all_reduce(mean)  # SUM: gloo has no AVG
        mean /= world
    return float(mean)


def train(args, *, rank: int, world: int, device, process_group,
          dual_factory=None) -> dict:
    """Run ``args.steps`` steps; return per-step rows plus a summary.

    :param dual_factory: ``(m, *, device, process_group) -> DualOptimizer``, overriding
        ``METHODS[args.method]`` (``--method`` then only labels the row). Used by
        ``paper/tune.py``.
    """
    set_seed(args.seed)
    amp_dtype = getattr(torch, args.dtype) if device.type == "cuda" else None

    shard = tokens_mod.TokenShard(args.tokens, args.seq_len)
    model = build_model(args.model, device=device, seed=args.seed,
                        vocab_size=args.vocab_size)
    model.train()  # from_pretrained returns eval mode

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

    # gates are attached before the DDP wrap so their parameters are synchronised too
    wrapped = model
    if world > 1:
        wrapped = nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    # Gates get their own, larger step (Adam moves ~lr per step; log_alpha has to travel
    # ~4) and no weight decay (it would pull density toward ~0.83).
    gate_ids = set() if gates is None else {id(p) for p in gates.gate_parameters()}
    groups = [{"params": [p for p in wrapped.parameters() if id(p) not in gate_ids],
               "lr": args.lr}]
    if gate_ids:
        groups.append({"params": [p for p in wrapped.parameters() if id(p) in gate_ids],
                       "lr": args.gate_lr, "weight_decay": 0.0})
    primal = torch.optim.AdamW(groups, lr=args.lr)

    dual = None
    if dual_factory is not None or METHODS[args.method] is not None:
        if gates is None:
            raise SystemExit(f"--method {args.method} needs gates; drop --no-gates")
        if dual_factory is not None:
            dual = dual_factory(m, device=device, process_group=process_group)
        else:
            dual = build_dual(args.method, m, device=device, process_group=process_group,
                              overrides=parse_dual_opts(args.dual_opt))

    # Same seed on every rank = shared gate noise, so `world x B` matches `1 x world*B`.
    gate_gen = None
    if gates is not None:
        gate_gen = torch.Generator(device=device)
        gate_gen.manual_seed(args.seed if args.sync_gates else args.seed + 9973 * rank)

    # train | val | test as fractions of the shard, so the split does not move with
    # world or batch size. Only val may be used for selection.
    n_val = int(len(shard) * args.val_frac)
    n_test = int(len(shard) * args.test_frac)
    n_train = len(shard) - n_val - n_test
    if n_train <= 0:
        raise SystemExit(
            f"val_frac={args.val_frac} + test_frac={args.test_frac} leaves no training "
            f"blocks in a shard of {len(shard)}"
        )
    val_start, test_start = n_train, n_train + n_val
    sampler = tokens_mod.BlockSampler(
        n_train, args.batch_size, rank=rank, world=world, seed=args.seed
    )
    tokens_per_step = args.batch_size * args.seq_len * world

    def eval_point(step: int, *, with_test: bool) -> dict:
        """One curve point at the median gates; uses no RNG, so training is unaffected."""
        if gates is not None:
            gates.use_median()
        point = {"step": step, "tokens_seen": (step + 1) * tokens_per_step}
        splits = [("val", val_start, n_val)] + ([("test", test_start, n_test)]
                                                if with_test else [])
        for name, start, count in splits:
            loss = evaluate(
                model, shard, start, count, batch_size=args.batch_size, device=device,
                rank=rank, world=world, amp_dtype=amp_dtype, max_batches=args.eval_batches,
            )
            if loss is not None:
                point[f"{name}_loss"] = loss
                point[f"{name}_ppl"] = float(np.exp(loss))
        return point

    comm = comm_accounting(model, m, world)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)  # per configuration: sweep.py reuses the process
    rows, eval_rows = [], []
    for step, indices in enumerate(sampler.stream(args.steps)):
        timer = StepTimer(device)
        timer.mark("start")

        ids = shard.batch(indices, device=device)
        if gates is not None:
            gates.resample(gate_gen)  # fresh draw per step, before the forward pass

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = wrapped(input_ids=ids, labels=ids).loss
        timer.mark("forward")

        if dual is not None:
            objective = dual.forward_update(
                loss, gates.constraints(args.eps, args.granularity))
        elif args.method == "penalty":
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

        row = {"step": step, "loss": float(loss.detach()), **timer.finish()}
        if gates is not None:
            with torch.no_grad():
                density = gates.densities(args.granularity)
            row["density_mean"] = float(density.mean())
            row["density_min"] = float(density.min())
            row["density_max"] = float(density.max())
            row["max_violation"] = float((density - args.eps).max())
        if dual is not None:
            duals = dual.duals.detach()
            row["dual_mean"] = float(duals.mean())
            row["dual_max"] = float(duals.max())
        rows.append(row)

        if rank == 0 and (step % max(1, args.steps // 10) == 0 or step == args.steps - 1):
            note = f"  density={row['density_mean']:.4f}" if gates is not None else ""
            print(f"  step {step:>5d}  loss={row['loss']:.4f}"
                  f"  {row['ms_total']:.1f} ms{note}", flush=True)
        # outside the timer; the last step is evaluated below, with test
        if args.eval_every and step % args.eval_every == 0 and step != args.steps - 1:
            eval_rows.append(eval_point(step, with_test=args.test_curve))

    warmup = min(args.warmup, max(0, len(rows) - 1))
    steady = rows[warmup:] or rows
    seconds = sum(r["ms_total"] for r in steady) / 1e3
    summary = {
        "method": args.method,
        "dual_opts": ",".join(args.dual_opt),
        "model": args.model,
        "world": world,
        "granularity": args.granularity if gates is not None else "none",
        "m": m,
        "eps": args.eps,
        "per_rank_batch": args.batch_size,
        "seq_len": args.seq_len,
        "steps": len(rows),
        "warmup_steps_dropped": warmup,
        "tokens_per_s": tokens_per_step * len(steady) / seconds if seconds else 0.0,
        "ms_per_step": seconds * 1e3 / len(steady),
        "final_loss": rows[-1]["loss"],
        **{k: sum(r[k] for r in steady) / len(steady)
           for k in ("ms_forward", "ms_backward", "ms_dual", "ms_opt")},
        **facts,
        **comm,
    }
    if args.method == "penalty":
        summary["penalty"] = args.penalty

    # The shipped model is the median gate (their App. A.1); report the density it
    # actually realises, so perplexity and density describe the same network.
    report = []
    if gates is not None:
        report = gates.median_report()
        summary["final_median_density"] = (
            sum(r["params_active"] for r in report) / sum(r["params_total"] for r in report)
        )
        for k in ("density_mean", "density_min", "density_max", "max_violation"):
            summary[f"final_{k}"] = rows[-1][k]

    # headline: final iterate on val and test, always
    final = eval_point(len(rows) - 1, with_test=True)
    if args.eval_every:
        eval_rows.append(final)
    for key in ("val_loss", "val_ppl", "test_loss", "test_ppl"):
        if key in final:
            summary[key] = final[key]
    summary["val_blocks"], summary["test_blocks"] = n_val, n_test
    scored = [r for r in eval_rows if "val_ppl" in r]
    if scored:
        best = min(scored, key=lambda r: r["val_ppl"])
        summary["best_val_ppl"] = best["val_ppl"]
        summary["best_val_step"] = best["step"]
    if device.type == "cuda":
        summary["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)

    return {"rows": rows, "summary": summary, "median_report": report,
            "eval_rows": eval_rows, "names": names}


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


def _pick_device(local_rank: int, world: int = 1) -> torch.device:
    """CUDA if every rank gets its own GPU, else CPU everywhere. Decided on ``world`` so
    all ranks agree (mixed nccl/gloo would deadlock)."""
    if torch.cuda.is_available() and world <= torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _ensure_tokens(args) -> None:
    """Write a synthetic shard if asked and none exists."""
    path = Path(args.tokens)
    if path.exists() or not args.synthetic:
        return
    vocab = args.vocab_size or 256
    n = max(args.steps + 8, 64) * args.batch_size * args.seq_len * 4
    written = tokens_mod.write_synthetic_shard(path, n, vocab, seed=args.seed)
    print(f"wrote a synthetic shard: {written} tokens -> {path}")


def build_parser() -> argparse.ArgumentParser:
    """The single-run flags; the drivers start from ``build_parser().parse_args([])``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub",
                        help="'stub' for the transformers-free stand-in, else a HF id")
    parser.add_argument("--tokens", default=None, help="path to a uint16 token shard")
    parser.add_argument("--synthetic", action="store_true",
                        help="write a random-token shard if --tokens does not exist")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--method", default="alm_gda_restart", choices=sorted(METHODS))
    parser.add_argument("--granularity", default="block",
                        choices=sparsity_gates.GRANULARITIES)
    parser.add_argument("--eps", type=float, default=0.5, help="target density")
    parser.add_argument("--penalty", type=float, default=1.0,
                        help="fixed penalty coefficient, for --method penalty")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5,
                        help="steps dropped before the throughput average")
    parser.add_argument("--eval-batches", type=int, default=16,
                        help="batches per rank per evaluation; 0 means the whole split")
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--test-frac", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="steps between curve points; 0 evaluates only at the end")
    parser.add_argument("--no-test-curve", dest="test_curve", action="store_false",
                        help="curve on val only; test is still measured at the end")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4, help="per rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2,
                        help="step size for the gate parameters")
    parser.add_argument("--dual-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="override a dual hyperparameter; repeatable, "
                             "e.g. --dual-opt kp=0.5 --dual-opt nu=0.1")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16",
                        help="autocast dtype on GPU; parameters are always float32")
    parser.add_argument("--init-open", type=float, default=0.95)
    parser.add_argument("--no-gates", action="store_true",
                        help="the plain LM: no hooks, no constraints")
    parser.add_argument("--no-mlp-gates", action="store_true")
    parser.add_argument("--no-head-gates", action="store_true")
    parser.add_argument("--no-sync-gates", dest="sync_gates", action="store_false",
                        help="draw independent gate samples per rank")
    parser.add_argument("--no-process-group", action="store_true",
                        help="skip the constraint all-reduce, to measure its redundancy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ranks", type=int, default=1,
                        help="re-exec under torchrun with this many ranks")
    parser.add_argument("--tag", default=None, help="suffix for the artifact names")
    parser.add_argument("--quick", action="store_true")
    return parser


def finalize(args):
    """Apply ``--quick``, fill in the shard path, reject impossible combinations."""
    if args.quick:
        args.steps = min(args.steps, 6)
        args.seq_len = min(args.seq_len, 32)
        args.batch_size = min(args.batch_size, 2)
        args.warmup = 1
        args.eval_batches = min(args.eval_batches, 2)
        args.eval_every = min(args.eval_every, 2) if args.eval_every else 0
        args.synthetic = True
    if args.tokens is None:
        args.tokens = str(RESULTS / EXPERIMENT / ("tokens_quick.bin" if args.quick
                                                  else "tokens.bin"))
    if args.method == "penalty" and args.no_gates:
        raise SystemExit("--method penalty needs gates")
    return args


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = finalize(build_parser().parse_args(argv))
    rank, world, local_rank = _dist_env()

    # Under torchrun: be a worker.
    if rank >= 0:
        device = _pick_device(local_rank, world)
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

    _ensure_tokens(args)

    # More than one rank asked for: re-exec under torchrun with the same argv
    # (workers take the branch above, so --ranks is ignored there).
    if args.ranks > 1:
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={args.ranks}",
            str(Path(__file__).resolve()), *argv,
        ]
        print(f"launching {args.ranks} ranks ...", flush=True)
        completed = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1"))
        if completed.returncode != 0:
            raise SystemExit(f"torchrun failed with code {completed.returncode}")
        return

    device = _pick_device(0)
    result = train(args, rank=0, world=1, device=device, process_group=None)
    _write(args, result, 1)


def _slug(args, world: int) -> str:
    """Artifact name. ``world`` comes from the live process group, not ``args.ranks``.
    scaling.py rebuilds this name to find the summary, so keep the two in step."""
    if args.method == "penalty":
        base = f"penalty_{args.granularity}_lam{args.penalty:g}_w{world}"
    else:
        base = f"{args.method}_{args.granularity}_eps{args.eps:g}_w{world}"
    if args.no_gates:
        base += "_nogates"
    if args.no_process_group and world > 1:
        base += "_nopg"
    return f"{base}_{args.tag}" if args.tag else base


def _write(args, result: dict, world: int) -> None:
    slug = _slug(args, world)
    write_csv(result["rows"], f"e3_steps_{slug}", EXPERIMENT)
    target = (f"penalty={args.penalty:g}" if args.method == "penalty"
              else f"eps={args.eps:g}")
    write_table([result["summary"]], f"e3_summary_{slug}", EXPERIMENT,
                title=f"E3: {args.method} on {args.model}, "
                      f"{args.granularity} granularity, {target}")
    if result["eval_rows"]:
        write_csv(result["eval_rows"], f"e3_eval_{slug}", EXPERIMENT)
    if result["median_report"]:
        write_table(result["median_report"], f"e3_density_{slug}", EXPERIMENT,
                    title="E3: per-group density at the median gates")
    _make_figure(result, slug)
    print(json.dumps(result["summary"], indent=2, default=str))


def _make_figure(result: dict, slug: str) -> None:
    rows = result["rows"]
    if len(rows) < 2 or "density_mean" not in rows[0]:
        return
    curve = [r for r in result["eval_rows"] if "val_ppl" in r]
    fig, axes, plt = figure(1, 3 if curve else 2, row_height=2.2)
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
    if curve:
        axes[2].plot([r["step"] for r in curve], [r["val_ppl"] for r in curve],
                     lw=1.0, marker="o", ms=2, label="val")
        held = [r for r in curve if "test_ppl" in r]
        if held:
            axes[2].plot([r["step"] for r in held], [r["test_ppl"] for r in held],
                         lw=1.0, marker="s", ms=2, label="test")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("perplexity (median gates)")
        axes[2].legend(fontsize="x-small")
    fig.tight_layout()
    save_figure(fig, f"e3_{slug}", EXPERIMENT)
    plt.close(fig)


if __name__ == "__main__":
    main()
