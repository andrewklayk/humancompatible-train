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


_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _expand_acc(df):
    """Expand the per-class ``acc`` array-string column into numeric acc_0..acc_{K-1}.

    train.py logs per-class accuracy as a stringified numpy array (e.g.
    ``"[0.45 0.31 ...]"``, possibly wrapped over several lines). That column is
    non-numeric, so the generic aggregation below skips it. Expanding it into one
    numeric column per class lets ``acc_j`` be seed-averaged like any other metric
    (-> ``acc_j_mean`` / ``acc_j_std`` in the aggregated CSV). No-op if absent or
    already numeric (e.g. a scalar-accuracy task)."""
    if "acc" not in df.columns or pd.api.types.is_numeric_dtype(df["acc"]):
        return df
    parsed = df["acc"].apply(lambda s: [float(x) for x in _NUM_RE.findall(str(s))])
    K = max((len(v) for v in parsed), default=0)
    if K == 0:
        return df.drop(columns=["acc"])
    acc_cols = {f"acc_{j}": [v[j] if j < len(v) else float("nan") for v in parsed]
                for j in range(K)}
    return df.drop(columns=["acc"]).assign(**acc_cols)


def _row_max_viol(df, cc, filt):
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


def _aggregate_split(runs, split, filt, expand_acc=False):
    """Seed-average one split's per-epoch curve over the (fold, init_seed) grid.

    Metric-agnostic: every numeric column (loss, max_viol, each c_i, and the opt-split
    KKT metrics grad_norm/L/compl/max_viol/lambda_i) gets ``_mean``, ``_std``, and
    the ``_std_fold`` / ``_std_init`` variance components. Returns (DataFrame,
    constraint_cols) or (None, []). ``expand_acc`` (CIFAR only) turns the per-class
    ``acc`` array-string column into numeric acc_j columns so accuracy aggregates too.
    """
    frames, cc = [], []
    for r in runs:
        p = os.path.join(r["dir"], f"{split}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if expand_acc:
            df = _expand_acc(df)
        cc = _c_cols(df) or cc
        if cc:
            df = df.assign(max_viol=_row_max_viol(df, cc, filt))
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


def _aggregate_cell(configs, filt, expand_acc=False):
    """Per-config aggregates for one cell (i.e. task-data-method), ordered by hyperparameter signature.

    Each item: {index, signature, meta, splits{split: DataFrame}, m, sel_split}.
    Selection split is 'val' (ml mode) or 'opt' (opt/train-only); configs with
    neither are dropped. ``expand_acc`` (CIFAR only) is forwarded to _aggregate_split.
    """
    items = []
    for i, sig in enumerate(sorted(configs)):
        info = configs[sig]
        splits, m = {}, 0
        for split in ("train", "val", "test", "opt"):
            res, cc = _aggregate_split(info["runs"], split, filt, expand_acc=expand_acc)
            if res is not None:
                splits[split], m = res, max(m, len(cc))
        sel = "val" if "val" in splits else "opt" if "opt" in splits else None
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
    ap.add_argument("--runs", default="multirun2/cifar10/", help="multirun root to scan recursively")
    ap.add_argument("--out", default="selection/", help="output dir (aggregated/ is created under it)")
    ap.add_argument("--approach", default="opt", choices=["ml", "opt"],
                    help="only aggregate runs of this approach; required if the tree mixes "
                         "approaches (ml and opt must not share a cell -- use a separate --out).")
    args = ap.parse_args()
    agg_dir = os.path.join(args.out, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    # cell (task, data, algorithm) -> signature -> {meta, runs:[{dir, fold, init_seed}]}
    cells, approaches = {}, set()
    # gets into each individual training run folder at "multirun/{task}/{algorithm}/{approach}/fold{K}_init{K}/{job_name}"
    # and groups by (task, data, algorithm)
    for meta_path in glob.glob(os.path.join(args.runs, "**", "run_meta.json"), recursive=True):
        with open(meta_path) as f:
            meta = json.load(f)
        appr = meta.get("approach", "ml")
        if args.approach is not None and appr != args.approach:
            continue
        approaches.add(appr) # check if we're mixing ML and OPT runs (if len > 1 by the end, then we are -> stop)
        cell = (meta["task"], meta.get("data"), meta["algorithm"])
        sig = json.dumps(meta["hyperparameters"], sort_keys=True, default=str)
        # idk why this is so convoluted but entry is that second {"meta": meta, runs: []} dict
        # clinically insane one-line way to update a (task, data, algorithm) "cell" with new runs OR initialize it if this is the first run
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
    # start aggregating: actual trajectory CSVs are read inside helper f-ns
    for cell, configs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        filt = next(iter(configs.values()))["meta"]["select_filter"]
        # Per-class accuracy curves are only wanted for the image tasks.
        expand_acc = cell[1] in ("cifar10", "cifar100")
        items = _aggregate_cell(configs, filt, expand_acc=expand_acc)
        if not items:
            print(f"[skip] {_cell_name(cell)}: no csv")
            continue
        _write_aggregated(items, cell, agg_dir)
        n_cfg += len(items)
        print(f"[agg]  {_cell_name(cell)}: {len(items)} configs")

    print(f"\nWrote {n_cfg} config aggregates to {agg_dir}/")


if __name__ == "__main__":
    main()
