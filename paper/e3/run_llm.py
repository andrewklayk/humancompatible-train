"""
E3 — sparsity-constrained LM fine-tuning, single GPU.

Implements Eq. (3) of Gallego-Posada, Ramirez, Erraqabi, Bengio & Lacoste-Julien,
*Controlled Sparsity via Constrained Optimization* (NeurIPS 2022, arXiv:2208.04425):
structured hard-concrete L0 gates on a decoder LM, with per-block or model-wide
constraints on the expected parameter density. The gates and constraints live in
:mod:`paper.problems.sparsity.sparse_lm`; this script is the training loop, the instrumentation
and the artifact writing.

**This file is one run.** Grids (method x eps x seed, hyperparameters) are Hydra multiruns
of ``paper/tune.py``, which builds each run from :func:`build_parser`'s defaults. Dual
optimizer settings come only from ``paper/conf/algorithm/<method>.yaml``.

**What E3 asks.** Do the dual methods deliver a requested density, what is the model worth
at that density (perplexity at the median gates, on held-out blocks), and what does the
dual layer cost per step.

Usage::

    # CPU smoke, no downloads and no `transformers` needed
    python paper/e3/run_llm.py --model stub --synthetic --quick

    python paper/e3/run_llm.py --model Qwen/Qwen2.5-0.5B \
        --tokens paper/results/e3/tokens.npy \
        --method alm_gda_restart --granularity block --eps 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from itertools import chain, islice, repeat
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper._harness import (
    ALGORITHMS,
    RESULTS,
    algorithm_factory,
    figure,
    save_figure,
    set_seed,
    write_csv,
    write_table,
)

EXPERIMENT = "e3"


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #


class StepTimer:
    """Phase timings for one step, honest on GPU as well as CPU.

    On CUDA every mark is a ``torch.cuda.Event`` and the elapsed times are read after a
    single ``synchronize()`` at the end of the step.
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
        out["ms_total"] = sum(out.values())
        return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def build_model(name: str, *, device, seed: int, vocab_size: int = None):
    """The stand-in model, or a HuggingFace causal LM, in float32.

    Parameters are float32 whatever ``--dtype`` says; that flag picks the *autocast*
    dtype. Holding master weights in bfloat16 drops a large share of the AdamW updates
    outright -- at ``lr=1e-4`` on weights of magnitude 1e-2 the update is around one
    bf16 ULP -- and the moment buffers would be bfloat16 too.

    ``transformers`` is imported lazily so the stub path needs no transformers.
    """
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


def _to_ids(blocks) -> torch.Tensor:
    return torch.from_numpy(np.stack(blocks).astype(np.int64))


def evaluate(model, blocks, *, batch_size, device, amp, max_batches=None):
    """Mean cross-entropy over a split's blocks, at whatever gates are installed.

    Returns ``None`` when the split is smaller than one batch, which is the ``--quick``
    case. ``max_batches`` caps the batches, always taking the split's first blocks, so
    every point on a curve is measured on identical data. ``None`` or ``0`` means the
    whole split.
    """
    loader = DataLoader(blocks, batch_size=batch_size, drop_last=True, collate_fn=_to_ids)
    batches = islice(loader, max_batches or None)
    total, n = torch.zeros((), device=device), 0
    with torch.no_grad():
        for ids in batches:
            ids = ids.to(device)
            with amp():
                total += model(input_ids=ids, labels=ids).loss.float()
            n += 1
    return float(total / n) if n else None


def train(args, *, device, dual_factory=None) -> dict:
    """Run ``args.steps`` steps; return per-step rows plus a summary.

    :param dual_factory: ``(m, *, device) -> DualOptimizer`` (see
        :func:`paper._harness.dual_factory`), or ``None`` for no duals. ``args.method``
        only labels the run.
    """
    set_seed(args.seed)
    amp_dtype = getattr(torch, args.dtype) if device.type == "cuda" else None

    def autocast():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    tokens = np.load(args.tokens, mmap_mode="r")
    blocks = tokens[: len(tokens) // args.seq_len * args.seq_len].reshape(-1, args.seq_len)
    model = build_model(
        args.model, device=device, seed=args.seed, vocab_size=args.vocab_size,
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
    m = 0 if gates is None else gates.m(args.granularity)

    # Weights get AdamW; gate parameters (log_alpha) get plain SGD at their own step size,
    # with no weight decay -- as in run_cifar.py.
    gate_ids = set() if gates is None else {id(p) for p in gates.gate_parameters()}
    optimizers = [torch.optim.AdamW(
        [p for p in model.parameters() if id(p) not in gate_ids], lr=args.lr)]
    if gates is not None:
        optimizers.append(torch.optim.SGD(gates.gate_parameters(), lr=args.gate_lr))
    dual = None
    if dual_factory is not None:
        if gates is None:
            raise SystemExit(f"--method {args.method} needs gates; drop --no-gates")
        dual = dual_factory(m, device=device)

    gate_gen = None
    if gates is not None:
        gate_gen = torch.Generator(device=device)
        gate_gen.manual_seed(args.seed)

    n_val = int(len(blocks) * args.val_frac)
    n_test = int(len(blocks) * args.test_frac)
    n_train = len(blocks) - n_val - n_test
    if n_train < args.batch_size:
        raise SystemExit(
            f"val_frac={args.val_frac} + test_frac={args.test_frac} leaves fewer than one "
            f"training batch in a shard of {len(blocks)} blocks"
        )
    split = {"val": blocks[n_train:n_train + n_val], "test": blocks[n_train + n_val:]}
    loader = DataLoader(
        blocks[:n_train], batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=_to_ids, generator=torch.Generator().manual_seed(args.seed),
    )
    # Reshuffled every epoch; one epoch is ~2 at the default 5000 steps.
    batches = islice(chain.from_iterable(repeat(loader)), args.steps)
    tokens_per_step = args.batch_size * args.seq_len

    def eval_point(step: int, *, with_test: bool) -> dict:
        """One curve point at the median gates; uses no RNG, so training is unaffected."""
        if gates is not None:
            gates.use_median()
        point = {"step": step, "tokens_seen": (step + 1) * tokens_per_step}
        for name in ("val", "test") if with_test else ("val",):
            loss = evaluate(
                model, split[name], batch_size=args.batch_size, device=device,
                amp=autocast, max_batches=args.eval_batches,
            )
            if loss is not None:
                point[f"{name}_loss"] = loss
                point[f"{name}_ppl"] = float(np.exp(loss))
        return point

    eval_rows = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for step, ids in enumerate(batches):
        timer = StepTimer(device)
        timer.mark("start")

        ids = ids.to(device)
        if gates is not None:
            gates.resample(gate_gen)  # fresh draw per step, before the forward pass

        with autocast():
            loss = model(input_ids=ids, labels=ids).loss
        timer.mark("forward")

        objective = loss if dual is None else dual.forward_update(
            loss, gates.constraints(args.eps, args.granularity))
        timer.mark("dual")

        objective.backward()
        timer.mark("backward")

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        timer.mark("opt")

        row = {
            "step": step,
            "loss": float(loss.detach()),
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
            duals = dual.duals.detach()
            row["dual_mean"] = float(duals.mean())
            row["dual_max"] = float(duals.max())
        rows.append(row)
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            note = f"  density={row['density_mean']:.4f}" if gates else ""
            print(f"  step {step:>5d}  loss={row['loss']:.4f}"
                  f"  {row['ms_total']:.1f} ms{note}", flush=True)
        # Outside the StepTimer region, so `tokens_per_s` is unaffected.
        if args.eval_every and step % args.eval_every == 0:
            eval_rows.append(eval_point(step, with_test=args.test_curve))

    warmup = min(args.warmup, max(0, len(rows) - 1))
    steady = rows[warmup:]
    seconds = sum(r["ms_total"] for r in steady) / 1e3
    summary = {
        "method": args.method,
        "model": args.model,
        "granularity": args.granularity if gates else "none",
        "m": m,
        "eps": args.eps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": len(rows),
        "warmup_steps_dropped": warmup,
        "tokens_per_s": tokens_per_step * len(steady) / seconds if seconds else 0.0,
        "ms_per_step": sum(r["ms_total"] for r in steady) / len(steady),
        "final_loss": rows[-1]["loss"],
        **{k: sum(r[k] for r in steady) / len(steady)
           for k in ("ms_forward", "ms_backward", "ms_dual", "ms_opt")},
        **sparse_lm.describe(model, gates),
    }
    if args.method == "penalty":
        summary["penalty"] = args.penalty

    # The shipped model is the median gate (their App. A.1); report the density it
    # actually realises, so perplexity and density describe the same network.
    report = []
    if gates is not None:
        report = gates.median_report()
        # What the *shipped* network realises, parameter-counted the way the constraint
        # is. `final_density_mean` is the expected density the constraint acts on, while
        # the perplexity below is measured at the median gates
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

    if gates is not None:
        summary.update({f"final_{k}": rows[-1][k] for k in (
            "density_mean", "density_min", "density_max", "max_violation")})
    if device.type == "cuda":
        summary["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)

    return {"rows": rows, "summary": summary, "median_report": report,
            "eval_rows": eval_rows}


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_tokens(args) -> None:
    """Write a synthetic shard if asked and none exists."""
    path = Path(args.tokens)
    if path.exists() or not args.synthetic:
        return
    vocab = args.vocab_size or 256
    n = max(args.steps + 8, 64) * args.batch_size * args.seq_len * 4
    ids = np.random.default_rng(args.seed).integers(0, vocab, n).astype(np.uint32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:  # a file handle, so np.save keeps the name as given
        np.save(f, ids)
    print(f"wrote a synthetic shard: {n} tokens -> {path}")


def build_parser() -> argparse.ArgumentParser:
    """The single-run flags. ``tune.py`` builds its runs from ``build_parser().parse_args([])``,
    so a default added here reaches it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub",
                        help="'stub' for the transformers-free stand-in, else a HF id")
    parser.add_argument("--tokens", default=None, help="path to a .npy token shard")
    parser.add_argument("--synthetic", action="store_true",
                        help="write a random-token shard if --tokens does not exist")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--method", default="alm_gda_restart", choices=ALGORITHMS,
                        help="a paper/conf/algorithm/*.yaml name")
    parser.add_argument("--granularity", default="block",
                        choices=sparsity_gates.GRANULARITIES)
    parser.add_argument("--eps", type=float, default=0.5, help="target density")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=5,
                        help="steps dropped before the throughput average")
    parser.add_argument("--eval-batches", type=int, default=16,
                        help="batches per evaluation; 0 means the whole split")
    parser.add_argument("--val-frac", type=float, default=0.01,
                        help="fraction of the shard held out for validation")
    parser.add_argument("--test-frac", type=float, default=0.01,
                        help="fraction of the shard held out for test; nothing selects "
                             "on it")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="steps between curve points; 0 evaluates only at the end")
    parser.add_argument("--no-test-curve", dest="test_curve", action="store_false",
                        help="curve on val only; test is still measured at the end")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2,
                        help="step size for the gate parameters; see train()")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16",
                        help="autocast dtype on GPU; parameters are always float32")
    parser.add_argument("--init-open", type=float, default=0.95)
    parser.add_argument("--no-gates", action="store_true",
                        help="the plain LM: no hooks, no constraints")
    parser.add_argument("--no-mlp-gates", action="store_true")
    parser.add_argument("--no-head-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="suffix for the artifact names")
    parser.add_argument("--quick", action="store_true")
    return parser


def finalize(args):
    """Apply ``--quick`` and fill in the shard path. Also called by ``tune.py``."""
    if args.quick:
        args.steps = min(args.steps, 6)
        args.seq_len = min(args.seq_len, 32)
        args.batch_size = min(args.batch_size, 2)
        args.warmup = 1
        args.eval_batches = min(args.eval_batches, 2)
        args.eval_every = min(args.eval_every, 2) if args.eval_every else 0
        args.synthetic = True
    if args.tokens is None:
        args.tokens = str(RESULTS / EXPERIMENT / ("tokens_quick.npy" if args.quick
                                                  else "tokens.npy"))
    return args


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = finalize(build_parser().parse_args(argv))

    _ensure_tokens(args)
    result = train(args, device=_pick_device(), dual_factory=algorithm_factory(args.method))
    _write(args, result)


def _slug(args) -> str:
    base = f"{args.method}_{args.granularity}_eps{args.eps:g}"
    if args.no_gates:
        base += "_nogates"
    return f"{base}_{args.tag}" if args.tag else base


def _write(args, result: dict) -> None:
    slug = _slug(args)
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
