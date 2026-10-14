"""
Fetch CIFAR into the directory E3's vision half trains from.

Run **once, on a login node** — compute nodes have no network, and both
``conf/experiment/e3_cifar.yaml`` and ``run_cifar.py`` default to ``download: false`` so
that a missing dataset fails loudly there instead of every array task racing to fetch it.
Same login-node/compute-node split as :mod:`paper.e3.prepare_data`, and the reason is the
same: a sweep that dies an hour in because an HTTP fetch failed is an expensive way to
learn that compute nodes are offline.

``--data`` must match ``experiment.data``, which is relative to the repository root
(``hydra.job.chdir: false``), so the default lands where the sweep looks.

Usage::

    python paper/e3/prepare_cifar.py                      # cifar10 into ./data
    python paper/e3/prepare_cifar.py --dataset cifar100
"""

from __future__ import annotations

import argparse
from pathlib import Path

# The dataset names run_cifar.STATS knows, hardcoded so this stays importable with no
# torch on the path -- the whole point is to run it before the training environment.
DATASETS = ("cifar10", "cifar100")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="cifar10", choices=DATASETS)
    parser.add_argument("--data", default="data", help="dataset root")
    args = parser.parse_args(argv)

    import torchvision

    dataset = getattr(torchvision.datasets, args.dataset.upper())
    for train in (True, False):
        split = dataset(args.data, train=train, download=True)
        print(f"  {args.dataset} {'train' if train else 'test':<5} {len(split):>6} images")
    print(f"-> {Path(args.data).resolve()}")


if __name__ == "__main__":
    main()
