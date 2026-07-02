"""Aggregate a multirun tree into per-config seed-averaged curves.

Stage 1 of 2 (stage 2 = select_best.py). Scans ``--runs`` for run_meta.json, groups
runs into (task, data, algorithm) cells, matches runs sharing the same
hyperparameters, and seed-averages their per-epoch curves over the (fold, init_seed)
grid -- for every split (train/val/test/opt) and every numeric metric, with the
``_std_fold`` / ``_std_init`` variance decomposition. Writes ONE tidy CSV (all
curves) + ONE JSON (metadata + hyperparameters) per cell under ``--out/aggregated/``.

These aggregated files are the single source of truth for BOTH selection
(select_best.py) and plotting (plotting/aggregate_results.py); neither re-reads the
raw runs.

ml and opt runs must not share a cell (different splits/metrics): filter with
``--approach`` and use a separate ``--out`` per approach (e.g. selection/opt).

Usage:
    python aggregate.py --runs multirun/ --out selection/ --approach ml
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

_NON_METRIC = {"epoch", "time", "fold", "init_seed"}


def _c_cols(df):
    """Constraint columns c_0, c_1, ... in numeric order."""
    return sorted((c for c in df.columns if re.fullmatch(r"c_\d+", c)),
                  key=lambda s: int(s.split("_")[1]))


def _row_maxc(df, cc, filt):
    """Per-row max constraint violation (signed for 'upper', absolute otherwise)."""
    return df[cc].max(axis=1) if filt == "upper" else df[cc].abs().max(axis=1)


def _cell_name(cell):
    return "_".join(str(x) for x in cell)


def _decompose(df, col):
    """Variance of ``col`` split across the (fold, init_seed) grid, per epoch.

    Returns (std_fold, std_init): std over folds of the per-fold means (DATA
    variance) and mean over folds of the per-fold init-std (OPTIMIZATION variance).
    ddof=0 so a single fold / init collapses to 0.0.
    """
    per_fold = df.groupby(["epoch", "fold"])[col]
    g = pd.DataFrame({"m": per_fold.mean(), "s": per_fold.std(ddof=0)}).reset_index()
    by_epoch = g.groupby("epoch")
    return by_epoch["m"].std(ddof=0), by_epoch["s"].mean()


def _aggregate_split(runs, split, filt):
    """Seed-average one split's per-epoch curve over the (fold, init_seed) grid.

    Metric-agnostic: every numeric column (loss, maxc, each c_i, and the opt-split
    KKT metrics grad_norm/L/compl/max_viol/lambda_i) gets ``_mean``, ``_std``, and
    the ``_std_fold`` / ``_std_init`` variance components. Returns (DataFrame,
    constraint_cols) or (None, []).
    """
    frames, cc = [], []
    for r in runs:
        p = os.path.join(r["dir"], f"{split}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cc = _c_cols(df) or cc
        if cc:
            df = df.assign(maxc=_row_maxc(df, cc, filt))
        frames.append(df.assign(fold=r["fold"], init_seed=r["init_seed"]))
    if not frames:
        return None, []
    allruns = pd.concat(frames, ignore_index=True)
    g = allruns.groupby("epoch")
    out = {"n_seeds": g.size()}
    for col in allruns.columns:
        if col in _NON_METRIC or not pd.api.types.is_numeric_dtype(allruns[col]):
            continue
        sf, si = _decompose(allruns, col)
        out[f"{col}_mean"], out[f"{col}_std"] = g[col].mean(), g[col].std(ddof=0)
        out[f"{col}_std_fold"], out[f"{col}_std_init"] = sf, si
    return pd.DataFrame(out).reset_index(), cc


def _aggregate_cell(configs, filt):
    """Per-config aggregates for one cell, ordered by hyperparameter signature.

    Each item: {index, signature, meta, splits{split: DataFrame}, m, sel_split}.
    Selection split is 'val' (ml mode) or 'train' (opt/train-only); configs with
    neither are dropped.
    """
    items = []
    for i, sig in enumerate(sorted(configs)):
        info = configs[sig]
        splits, m = {}, 0
        for split in ("train", "val", "test", "opt"):
            res, cc = _aggregate_split(info["runs"], split, filt)
            if res is not None:
                splits[split], m = res, max(m, len(cc))
        sel = "val" if "val" in splits else "train" if "train" in splits else None
        if sel is None:
            continue
        items.append({"index": i, "signature": sig, "meta": info["meta"],
                      "splits": splits, "m": m, "sel_split": sel})
    return items


def _write_aggregated(items, cell, agg_dir):
    """One tidy CSV + one metadata JSON per cell (no duplication between them):

      <cell>.csv   all curves, long: one row per (config, split, epoch); columns
                   config, split, epoch, n_seeds, <metric>_mean/std/std_fold/std_init
                   (union over splits -- NaN where a metric doesn't apply to a split).
      <cell>.json  {task, data, algorithm, bound, select_filter, configs:[{config_index,
                   sel_split, m, n_seeds, signature, hyperparameters}, ...]}
    """
    base = os.path.join(agg_dir, _cell_name(cell))

    frames = [df.assign(config=it["index"], split=split)
              for it in items for split, df in it["splits"].items()]
    long = pd.concat(frames, ignore_index=True)
    id_cols = ["config", "split", "epoch", "n_seeds"]
    long = long[id_cols + [c for c in long.columns if c not in id_cols]]
    long.to_csv(base + ".csv", index=False)

    meta = {
        "task": cell[0], "data": cell[1], "algorithm": cell[2],
        "bound": items[0]["meta"]["bound"],
        "select_filter": items[0]["meta"]["select_filter"],
        "configs": [{
            "config_index": it["index"], "sel_split": it["sel_split"], "m": it["m"],
            "n_seeds": int(it["splits"][it["sel_split"]]["n_seeds"].max()),
            "signature": it["signature"],
            "hyperparameters": it["meta"]["hyperparameters"],
        } for it in items],
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="multirun root to scan recursively")
    ap.add_argument("--out", default="selection", help="output dir (aggregated/ is created under it)")
    ap.add_argument("--approach", default=None, choices=["ml", "opt"],
                    help="only aggregate runs of this approach; required if the tree mixes "
                         "approaches (ml and opt must not share a cell -- use a separate --out).")
    args = ap.parse_args()
    agg_dir = os.path.join(args.out, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    # cell (task, data, algorithm) -> signature -> {meta, runs:[{dir, fold, init_seed}]}
    cells, approaches = {}, set()
    for meta_path in glob.glob(os.path.join(args.runs, "**", "run_meta.json"), recursive=True):
        with open(meta_path) as f:
            meta = json.load(f)
        appr = meta.get("approach", "ml")
        if args.approach is not None and appr != args.approach:
            continue
        approaches.add(appr)
        cell = (meta["task"], meta.get("data"), meta["algorithm"])
        sig = json.dumps(meta["hyperparameters"], sort_keys=True, default=str)
        entry = cells.setdefault(cell, {}).setdefault(sig, {"meta": meta, "runs": []})
        entry["runs"].append({"dir": os.path.dirname(meta_path),
                              "fold": int(meta.get("fold", 0)),
                              "init_seed": int(meta.get("init_seed", 0))})

    if args.approach is None and len(approaches) > 1:
        print(f"Runs under {args.runs} mix approaches {sorted(approaches)}; re-run with "
              f"--approach ml or --approach opt (and a per-approach --out, e.g. selection/opt).")
        return
    if not cells:
        print(f"No runs (run_meta.json) found under {args.runs}")
        return

    n_cfg = 0
    for cell, configs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        filt = next(iter(configs.values()))["meta"]["select_filter"]
        items = _aggregate_cell(configs, filt)
        if not items:
            print(f"[skip] {_cell_name(cell)}: no train or val csv")
            continue
        _write_aggregated(items, cell, agg_dir)
        n_cfg += len(items)
        print(f"[agg]  {_cell_name(cell)}: {len(items)} configs")

    print(f"\nWrote {n_cfg} config aggregates to {agg_dir}/")


if __name__ == "__main__":
    main()
