"""
Turn a ``paper/tune.py`` E2c sweep into ``paper/conf/e2c_tuned.yaml``.

Manual step 2 of E2c's tuning. Step 1 is the sweep itself::

    python paper/tune.py -m experiment=e2c_per_iteration +sweep=e2c_alm \\
        experiment.problem.dataset=income,dutch experiment.problem.per_group=4,8,16
    python paper/tune.py -m experiment=e2c_per_epoch +sweep=e2c_alm \\
        experiment.problem.dataset=income,dutch

which writes one ``row.json`` per job under ``paper/multirun/e2c/alm/``. This script
picks the lowest-objective job per (dataset, per_group, cadence) cell -- already
feasibility-first via ``tune.py``'s ``INFEASIBLE_WEIGHT`` penalty on train violation,
same as ``tune_report.py``'s ranking -- and writes the winners to a small committed
file that ``c_cadence.py`` reads at import time.

Usage::

    python paper/e2/tune_apply.py --runs paper/multirun/e2c/alm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paper.tune_report import load

CONF_PATH = Path(__file__).resolve().parents[1] / "conf" / "e2c_tuned.yaml"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="apply an E2c tuning sweep")
    parser.add_argument("--runs", required=True,
                        help="a paper/tune.py multirun directory, e.g. paper/multirun/e2c/alm")
    args = parser.parse_args(argv)

    rows = load(Path(args.runs))
    if not rows:
        raise SystemExit(f"no row.json under {args.runs} -- did the sweep run?")

    best = {}
    for row in rows:
        cfg = row["_config"]
        key = (cfg["experiment.problem.dataset"], cfg["experiment.problem.per_group"],
               cfg["experiment.cadence"])
        if key not in best or row["objective"] < best[key]["objective"]:
            best[key] = row

    lines = ["# Written by paper/e2/tune_apply.py -- do not hand-edit.", ""]
    for (dataset, per_group, cadence), row in sorted(best.items(), key=str):
        cfg = row["_config"]
        problem_name = f"{dataset}_{cfg['experiment.problem.shape']}"
        lines.append(f"{problem_name}_{per_group}_{cadence}:")
        lines.append(f"  primal_lr: {cfg['algorithm.primal.lr']}")
        lines.append(f"  dual_lr: {cfg['algorithm.dual.lr']}")
    CONF_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(best)} cell(s) -> {CONF_PATH}")


if __name__ == "__main__":
    main()
