"""
Rank the jobs of a ``paper/tune.py`` sweep.

Reads the ``row.json`` sidecar each job writes, not the Hydra configs -- the same
separation ``new_bench`` keeps between ``run.py`` and ``aggregate.py``, so the layout is
the only contract and a config change cannot break post-processing.

Usage::

    python paper/tune_report.py --runs paper/multirun/e2a/alm
    python paper/tune_report.py --runs paper/multirun --top 10 --out best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper._harness import set_results_dir, write_csv, write_table

# Config keys worth showing beside the metrics. A sweep varies a handful of them; the rest
# are constant across the grid and would be noise in the table.
SWEPT_PREFIXES = ("algorithm.primal.", "algorithm.dual.", "experiment.")


def _flatten(obj, prefix=""):
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``, so config keys can be compared as scalars."""
    out = {}
    for key, value in (obj or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def load(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("row.json")):
        record = json.loads(path.read_text())
        config = _flatten(record.pop("config", {}))
        record["job"] = str(path.parent.relative_to(root))
        rows.append({**record, "_config": config})
    return rows


def varying(rows: list[dict]) -> list[str]:
    """The config keys that actually differ across the sweep -- i.e. what was swept."""
    keys = {key for row in rows for key in row["_config"]
            if key.startswith(SWEPT_PREFIXES)}
    return sorted(
        key for key in keys
        if len({json.dumps(row["_config"].get(key), default=str) for row in rows}) > 1
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="rank a paper/tune.py sweep")
    parser.add_argument("--runs", required=True, help="a multirun or outputs directory")
    parser.add_argument("--top", type=int, default=20, help="rows to show; 0 = all")
    parser.add_argument("--out", default=None,
                        help="write <name>.csv/.md next to the runs instead of stdout only")
    args = parser.parse_args(argv)

    root = Path(args.runs)
    rows = load(root)
    if not rows:
        raise SystemExit(f"no row.json under {root} -- did the sweep run?")

    swept = varying(rows)
    # Failed jobs write no row.json at all, so anything here has an objective; inf is the
    # marker for a run that finished but produced nothing usable.
    rows.sort(key=lambda r: float(r.get("objective", float("inf"))))

    metrics = [key for key in rows[0]
               if key not in ("_config", "config", "job") and not key.startswith("_")]
    columns = ["job"] + swept + metrics
    table = [{**{key: row["_config"].get(key) for key in swept},
              **{key: row.get(key) for key in metrics},
              "job": row["job"]}
             for row in rows]

    shown = table if args.top <= 0 else table[:args.top]
    print(f"\n{len(rows)} runs under {root}; swept: {', '.join(swept) or '(nothing)'}\n")
    for rank, row in enumerate(shown, start=1):
        detail = "  ".join(f"{key.split('.')[-1]}={row[key]}" for key in swept)
        print(f"  {rank:>3}. objective {float(row['objective']):<12.5g} {detail}")

    if args.out:
        # Artifacts land beside the runs they summarize, not in paper/results/.
        set_results_dir(root)
        write_csv(table, args.out, "")
        write_table(shown, args.out, "", columns=columns, floatfmt="{:.5g}",
                    title=f"Sweep results under {root}, best {len(shown)} of {len(rows)}")


if __name__ == "__main__":
    main()
