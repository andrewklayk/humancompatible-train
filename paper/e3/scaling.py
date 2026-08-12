"""
E3 — what the dual layer costs in throughput, over the 1 -> 2 -> 4 GPU ladder.

Runs ``run_llm.py`` at each world size in a fresh ``torchrun`` job and reads back the
summary it writes. Two configurations per rung, **both with gates attached**:

* ``adam`` — the gated model with no constraint and no multipliers;
* ``alm_gda_restart`` — the same, plus the constraint and the dual update.

The difference between them is the dual layer. Timing a constrained run against the
*ungated* model would instead measure the gates, which is a property of the sparsity
formulation and not of any optimizer — the same control-design point E2a arrived at.
``run_llm.py`` times each phase with ``torch.cuda.Event`` around an explicit
``synchronize()``, so the breakdown is compute rather than kernel-launch time.

Communication is reported analytically from :func:`run_llm.comm_accounting`: the dual
all-reduce moves ``4m`` bytes against the gradient all-reduce's whole trainable footprint,
and the ratio is what makes the cost ``O(m)`` rather than ``O(n)``.

**No predictions are registered** — see ``run_llm.py``'s docstring.

Usage::

    python paper/e3/scaling.py --quick --worlds 1,2                   # CPU stub, gloo
    python paper/e3/scaling.py --model Qwen/Qwen2.5-0.5B --tokens .../tokens.bin
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from paper._harness import RESULTS, figure, save_figure, write_csv, write_table

EXPERIMENT = "e3"
RUNNER = Path(__file__).resolve().parent / "run_llm.py"

CONFIGS = (
    ("unconstrained", "adam"),
    ("constrained", "alm_gda_restart"),
)


def _run(args, method: str, world: int, tag: str) -> dict:
    """One ``run_llm.py`` job; returns the summary it wrote."""
    command = [
        sys.executable, str(RUNNER),
        "--model", args.model, "--method", method, "--granularity", args.granularity,
        "--eps", str(args.eps), "--steps", str(args.steps),
        "--seq-len", str(args.seq_len), "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--dual-lr", str(args.dual_lr), "--dtype", args.dtype,
        "--eval-batches", "0", "--seed", str(args.seed),
        "--ranks", str(world), "--tag", tag,
    ]
    if args.tokens:
        command += ["--tokens", args.tokens]
    if args.synthetic:
        command.append("--synthetic")
    if args.quick:
        command.append("--quick")

    print(f"\n=== world={world}  {method} ===", flush=True)
    completed = subprocess.run(command, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if completed.returncode != 0:
        raise SystemExit(f"{method} at world={world} failed ({completed.returncode})")

    slug = f"{method}_{args.granularity}_eps{args.eps:g}_w{world}_{tag}"
    path = RESULTS / EXPERIMENT / f"e3_summary_{slug}.json"
    if not path.exists():
        raise SystemExit(f"{path} was not written; the run produced no summary")
    return json.loads(path.read_text())[0]


def ladder(args) -> list[dict]:
    rows = []
    for world in args.worlds:
        for label, method in CONFIGS:
            summary = _run(args, method, world, tag=label)
            rows.append({"configuration": label, "world": world, **summary})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per rung: throughput, scaling efficiency, the dual layer's share."""
    by = {(r["configuration"], r["world"]): r for r in rows}
    worlds = sorted({r["world"] for r in rows})
    base = worlds[0]
    out = []
    for world in worlds:
        unc, con = by[("unconstrained", world)], by[("constrained", world)]
        row = {
            "world": world,
            "tok/s unconstrained": unc["tokens_per_s"],
            "tok/s constrained": con["tokens_per_s"],
            # What the dual layer costs, as a share of the gated model's throughput.
            "dual layer cost": 1.0 - con["tokens_per_s"] / unc["tokens_per_s"],
            "ms/step constrained": con["ms_per_step"],
            "ms dual": con["ms_dual"],
            "ms dual share": con["ms_dual"] / con["ms_per_step"],
            "m": con["m"],
            "dual bytes": con["dual_payload_bytes"],
            "grad bytes": con["grad_payload_bytes"],
            "dual/grad bytes": con["dual_over_grad"],
        }
        for label in ("unconstrained", "constrained"):
            ref = by[(label, base)]["tokens_per_s"]
            scale = by[(label, world)]["tokens_per_s"] / ref if ref else float("nan")
            row[f"speedup {label}"] = scale
            row[f"efficiency {label}"] = scale / (world / base)
        if "peak_mem_bytes" in con:
            row["peak mem (GiB)"] = con["peak_mem_bytes"] / 2**30
        out.append(row)
    return out


def make_figure(summary: list[dict]) -> None:
    fig, axes, plt = figure(1, 2, row_height=2.4)
    worlds = [r["world"] for r in summary]
    for key, label in (("tok/s unconstrained", "unconstrained"),
                       ("tok/s constrained", "constrained")):
        axes[0].plot(worlds, [r[key] for r in summary], marker="o", ms=3, label=label)
    ideal = summary[0]["tok/s unconstrained"]
    axes[0].plot(worlds, [ideal * w / worlds[0] for w in worlds],
                 color="k", ls="--", lw=0.8, label="ideal")
    axes[0].set_xlabel("GPUs")
    axes[0].set_ylabel("tokens/s")
    axes[0].legend(fontsize="x-small")

    axes[1].bar([str(w) for w in worlds], [r["ms dual share"] * 100 for r in summary])
    axes[1].set_xlabel("GPUs")
    axes[1].set_ylabel("dual update, % of step time")
    fig.tight_layout()
    save_figure(fig, "e3_scaling", EXPERIMENT)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="stub")
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--granularity", default="layer")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8, help="per rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--dual-lr", type=float, default=1e-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", default=None,
                        help="comma-separated ladder; default 1,2,4 capped at the "
                             "visible GPU count")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="accepted for parity; E3 registers no predictions")
    args = parser.parse_args(argv)

    if args.worlds:
        args.worlds = [int(w) for w in args.worlds.split(",")]
    else:
        available = max(1, torch.cuda.device_count())
        args.worlds = [w for w in (1, 2, 4) if w <= available] or [1]
    if args.quick:
        args.steps = 6

    rows = ladder(args)
    write_csv(rows, "e3_scaling_runs", EXPERIMENT)
    summary = summarize(rows)
    write_table(summary, "e3_scaling", EXPERIMENT,
                title="E3: throughput and the dual layer's share over the GPU ladder")
    make_figure(summary)
    print()
    for row in summary:
        print(f"  {row['world']} GPU  "
              f"{row['tok/s constrained']:>10.0f} tok/s constrained  "
              f"({row['dual layer cost'] * 100:+.2f}% vs unconstrained, "
              f"dual {row['ms dual share'] * 100:.2f}% of the step, "
              f"efficiency {row['efficiency constrained']:.2f})")


if __name__ == "__main__":
    main()
