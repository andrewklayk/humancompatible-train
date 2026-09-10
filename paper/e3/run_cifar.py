"""
E3 — controlled sparsity on CIFAR with a ResNet.

Implements Eq. (3) of Gallego-Posada, Ramirez, Erraqabi, Bengio & Lacoste-Julien,
*Controlled Sparsity via Constrained Optimization* (NeurIPS 2022, arXiv:2208.04425) on the
vision setting that paper reports: structured hard-concrete L0 gates on the intermediate
channels of every residual block, with a constraint on the expected parameter density,
model-wide or per block.

    min_{theta,phi}  E_z[ CE(f(x; theta * z), y) ]   s.t.   density_g(phi) <= eps_g

The gates and the density arithmetic live in :mod:`paper.problems.sparse_resnet` and
:mod:`paper.problems.sparse_lm`; this file is the training loop and the artifacts, and is
meant to be read top to bottom. Everything the run does to the model is in :func:`train`.

Three numbers per configuration:

* ``achieved - eps`` — did the method land on the density that was asked for. The
  constraint is a closed form in the gate parameters, with no data in it, so the **final**
  value is the achieved density: there is nothing to average over.
* ``test_acc`` — what the network is worth there, at the median (test-time) gates.
* ``median_density`` — what a purged network would actually keep, which is not the
  expected density: the median of a gate is not its open probability.

Baselines: ``--method none`` trains with gates installed but no sparsity pressure (they
stay near their 95%-open initialisation, so this is the dense reference at identical code
paths), and ``--method penalty`` is the penalized formulation of their Eq. (1)-(2), the
fixed-lambda baseline their Fig. 1 compares against.

Like the rest of E3 this registers no predictions; ``--check`` is accepted and exits 0.

Usage::

    python paper/e3/run_cifar.py --quick --synthetic
    python paper/e3/run_cifar.py --data ~/cifar_exp/data --method alm_gda_restart --eps 0.3
    python paper/e3/run_cifar.py --data ~/cifar_exp/data --epochs 100 \
        --method alm_gda_restart,alm_quad,nupi,pbm --eps 0.1,0.3,0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.problems.sparsity import sparse_lm, sparse_resnet, sparsity_gates
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paper._harness import figure, save_figure, set_seed, write_csv, write_table
from paper.e3 import run_llm
from paper.problems.sparsity import sparse_resnet

EXPERIMENT = "e3"

# The dual optimizers and their hyperparameters are shared with the LM half of E3, so the
# two settings compare the same configurations rather than two independent tunings.
METHODS = {
    "none": None,      # gates installed, no sparsity pressure: the dense reference
    # "penalty": None,   # their Eq. (1)-(2): loss + lambda * mean density
    **{name: build for name, build in run_llm.METHODS.items() if build is not None},
}

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def build_loaders(args):
    """Train and test loaders, with the standard crop/flip augmentation."""
    mean, std, n_classes = STATS[args.dataset]
    if args.synthetic:
        # Lets the whole path run with no dataset on disk, for smoke tests.
        def fake(n):
            generator = torch.Generator().manual_seed(args.seed)
            return TensorDataset(
                torch.randn(n, 3, 32, 32, generator=generator),
                torch.randint(n_classes, (n,), generator=generator),
            )

        train, test = fake(512), fake(256)
    else:
        import torchvision
        from torchvision import transforms

        normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        dataset = getattr(torchvision.datasets, args.dataset.upper())
        train = dataset(
            args.data, train=True, download=args.download,
            transform=transforms.Compose(
                [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
                + normalize
            ),
        )
        test = dataset(
            args.data, train=False, download=args.download,
            transform=transforms.Compose(normalize),
        )

    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                   num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0),
        DataLoader(test, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin),
        n_classes,
    )


@torch.no_grad()
def accuracy(model, loader, device, max_batches=None) -> float:
    """Top-1 at whatever gates are currently installed."""
    model.eval()
    correct = seen = 0
    for step, (x, y) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum())
        seen += y.numel()
    model.train()
    return correct / max(seen, 1)


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #


def train(args, *, label, method, eps, penalty, train_loader, test_loader, n_classes, device,
          dual_factory=None):
    """Train one configuration; return per-epoch rows plus a summary.

    :param dual_factory: ``(m, device, process_group, lr) -> DualOptimizer``, overriding
        ``METHODS[method]``. The registry fixes every dual hyperparameter but the step
        size, so this is what lets ``paper/tune.py`` sweep the rest.
    """
    set_seed(args.seed)
    model = sparse_resnet.cifar_resnet(args.depth, n_classes).to(device)
    gates = sparse_resnet.attach_gates(model, init_open=args.init_open, seed=args.seed,
                                       device=device)
    m = gates.m(args.granularity)

    # Weights get SGD, gate parameters get Adam at a much larger step and no weight decay:
    # carrying log_alpha from a 95%-open initialisation to a 10%-density solution is a move
    # of several units, which SGD at 0.1 with decay pulling it back would never make. The
    # split is also what arXiv:2208.04425 does.
    gate_ids = {id(p) for p in gates.gate_parameters()}
    primal = torch.optim.SGD(
        [p for p in model.parameters() if id(p) not in gate_ids],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True,
    ) 
    gate_optim = torch.optim.Adam(gates.gate_parameters(), lr=args.gate_lr)
    build = METHODS[method] if dual_factory is None else dual_factory
    dual = None if build is None else build(m, device, None, args.dual_lr)

    rows = []
    print("Starting")
    for epoch in range(args.epochs):
        started = time.perf_counter()
        total = seen = 0
        for step, (x, y) in enumerate(train_loader):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Fresh gate draw, before the forward: the sample is what carries gradients to
            # log_alpha, and reusing one across two backward passes raises from autograd.
            gates.resample()
            loss = F.cross_entropy(model(x), y)

            if dual is not None:
                objective = dual.forward_update(loss, gates.constraints(eps, args.granularity))
            elif method == "penalty":
                objective = loss + penalty * gates.densities(args.granularity).mean()
            else:
                objective = loss

            objective.backward()
            primal.step()
            gate_optim.step()
            primal.zero_grad(set_to_none=True)
            gate_optim.zero_grad(set_to_none=True)

            total += float(loss.detach()) * y.numel()
            seen += y.numel()
        # schedule.step()

        with torch.no_grad():
            density = gates.densities(args.granularity)
        # The test-time model is the median gate (their Appendix A.1), not a fresh sample.
        gates.use_median()
        report = gates.median_report()
        row = {
            "run": label,
            "method": method,
            "eps": eps,
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "test_acc": accuracy(model, test_loader, device, args.eval_batches),
            "density_mean": float(density.mean()),
            "density_min": float(density.min()),
            "density_max": float(density.max()),
            "max_violation": float((density - eps).max()),
            "median_density": sum(r["params_active"] for r in report)
            / sum(r["params_total"] for r in report),
            "seconds": time.perf_counter() - started,
        }
        if dual is not None:
            duals = dual.duals.detach()
            row["dual_mean"] = float(duals.mean())
            row["dual_max"] = float(duals.max())
        rows.append(row)
        print(f"  [{label}] epoch {epoch:>3d}  loss={row['train_loss']:.3f}"
              f"  acc={row['test_acc']:.3f}  density={row['density_mean']:.3f}"
              f"  ({row['seconds']:.1f}s)", flush=True)

    last = rows[-1]
    summary = {
        "run": label,
        "method": method,
        "eps": eps,
        "penalty": penalty,
        "granularity": args.granularity,
        "m": m,
        "final_density": last["density_mean"],
        "achieved_minus_eps": last["max_violation"],
        "median_density": last["median_density"],
        "test_acc": last["test_acc"],
        "best_test_acc": max(r["test_acc"] for r in rows),
        "epochs": len(rows),
        "s_per_epoch": sum(r["seconds"] for r in rows) / len(rows),
        **sparse_lm.describe(model, gates),
    }
    if device.type == "cuda":
        summary["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    return rows, summary, report


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #


def grid(args):
    """``(label, method, eps, penalty)`` — constrained methods sweep eps, penalty sweeps lambda."""
    for method in args.method:
        if method == "none":
            yield "none", method, float("nan"), 0.0
        elif method == "penalty":
            for lam in args.penalty:
                yield f"penalty@{lam:g}", method, float("nan"), lam
        else:
            for eps in args.eps:
                yield f"{method}@{eps:g}", method, eps, 0.0


def figures(rows, args, suffix):
    fig, axes, plt = figure(1, 2, row_height=2.4)
    for label in dict.fromkeys(r["run"] for r in rows):
        series = [r for r in rows if r["run"] == label]
        epochs = [r["epoch"] for r in series]
        axes[0].plot(epochs, [r["density_mean"] for r in series], lw=1.0, label=label)
        axes[1].plot(epochs, [r["test_acc"] for r in series], lw=1.0, label=label)
    for eps in args.eps:
        axes[0].axhline(eps, color="0.4", ls="--", lw=0.7)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("expected density")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy (median gates)")
    axes[1].legend(fontsize=5)
    fig.tight_layout()
    save_figure(fig, f"e3_cifar{suffix}", EXPERIMENT)
    plt.close(fig)


def _csv(convert):
    return lambda text: [convert(part) for part in text.split(",") if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="cifar10", choices=sorted(STATS))
    parser.add_argument("--data", default="data", help="dataset root")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="random images instead of CIFAR, for smoke runs")
    parser.add_argument("--method", type=_csv(str), default=["alm_gda_restart"],
                        help=f"comma-separated; one of {sorted(METHODS)}")
    parser.add_argument("--eps", type=_csv(float), default=[1.0, 0.3],
                        help="comma-separated target densities")
    parser.add_argument("--penalty", type=_csv(float), default=[0.1, 1.0],
                        help="comma-separated lambdas, for --method penalty")
    parser.add_argument("--granularity", default="model", choices=sparsity_gates.GRANULARITIES)
    parser.add_argument("--depth", type=int, default=18, help="ResNet depth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 = the whole set")
    parser.add_argument("--eval-batches", type=int, default=0, help="0 = the whole test set")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--dual-lr", type=float, default=1e-2)
    parser.add_argument("--init-open", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None, help="suffix for the artifact names")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="accepted for parity; E3 registers no predictions")
    return parser


def finalize(args):
    unknown = [name for name in args.method if name not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {sorted(METHODS)}")
    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.steps_per_epoch = args.steps_per_epoch or 3
        args.eval_batches = args.eval_batches or 2
        args.batch_size = min(args.batch_size, 32)
        args.workers = 0
    return args


def main(argv=None) -> None:
    args = finalize(build_parser().parse_args(argv))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_classes = build_loaders(args)
    suffix = f"_{args.tag}" if args.tag else ""

    rows, summaries, densities = [], [], []
    for label, method, eps, penalty in grid(args):
        print(f"\n=== {label} ===", flush=True)
        epochs, summary, report = train(
            args, label=label, method=method, eps=eps, penalty=penalty,
            train_loader=train_loader, test_loader=test_loader,
            n_classes=n_classes, device=device,
        )
        rows += epochs
        summaries.append(summary)
        densities += [dict(run=label, **r) for r in report]

    write_csv(rows, f"e3_cifar_epochs{suffix}", EXPERIMENT)
    write_csv(densities, f"e3_cifar_density{suffix}", EXPERIMENT)
    write_table(
        summaries, f"e3_cifar{suffix}", EXPERIMENT, floatfmt="{:.4f}",
        columns=["run", "eps", "final_density", "achieved_minus_eps", "median_density",
                 "test_acc", "best_test_acc", "s_per_epoch"],
        title=f"E3: {args.dataset}, ResNet-{args.depth}, {args.granularity} granularity",
    )
    figures(rows, args, suffix)
    print(json.dumps(summaries, indent=2, default=str))
    if args.check:
        print("\nE3 registers no predictions; --check has nothing to gate on.")


if __name__ == "__main__":
    main()
