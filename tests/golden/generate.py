"""
Regenerate the golden trajectories in ``tests/golden/trajectories/``.

Run this ONCE against the pre-refactor code, commit the result, and then do not
run it again unless a behavioural change is *intended* -- the whole point is that
the stored files are a record of how the optimizers behaved before the base-class
extraction.

Usage::

    python -m tests.golden.generate            # write every scenario
    python -m tests.golden.generate alm_basic  # write selected scenarios
"""

import sys
from pathlib import Path

import torch

from .scenarios import N_STEPS, SCENARIOS, run_scenario

DATA_DIR = Path(__file__).parent / "trajectories"


def main(names=None):
    names = names or sorted(SCENARIOS)
    DATA_DIR.mkdir(exist_ok=True)

    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}")
        traj = run_scenario(name)
        payload = {
            "trajectory": traj,
            "torch_version": torch.__version__,
            "n_steps": N_STEPS,
        }
        torch.save(payload, DATA_DIR / f"{name}.pt")
        keys = ",".join(sorted(traj))
        print(f"wrote {name}.pt  ({keys})")

    print(f"\n{len(names)} scenario(s) written to {DATA_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
