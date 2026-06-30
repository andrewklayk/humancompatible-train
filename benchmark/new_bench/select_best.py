"""Standalone best-config selection over a multirun output tree.

Decoupled from training. Aggregation is two-level:

  1. GROUP runs into "benchmark cells" by (task, data, algorithm).
  2. Within a cell, MATCH runs that share the same hyperparameters across seeds,
     and aggregate their per-epoch curves over seeds -> per (config, epoch) mean &
     std of loss, of the max constraint violation, AND of each individual signed
     constraint c_i -- for EVERY available split (train / val / test). Each such
     seed-averaged set of curves is SAVED (one file per config per cell) together
     with its hyperparameters, so selection AND plotting can be re-run / audited
     later without re-reading the raw runs. The aggregated files are the single
     source of truth consumed by plotting/aggregate_results.py.
  3. SELECT, on the seed-averaged VAL curves, the best config by a WINDOWED COLLAPSE
     (the shared `collapse`, also used by plotting/aggregate_results.py): each val curve
     is reduced to a representative (loss, violation) -- the mean over the last
     `--tail` epochs, or (with --rolling) the values at the epoch minimising the
     rolling-`tail` mean loss -- then the *feasible* config with the minimum loss is
     picked:
        filter='upper': keep configs with mean(max_j c_j)   <= bound * mult
        filter='both' : keep configs with mean(max_j |c_j|) <= bound * mult
        filter='none' : no feasibility filter (adam) -> argmin loss
     One selection is produced per feasibility-slack multiplier (--tols).

This picks the config that is best ON AVERAGE across seeds (not the luckiest
seed). Writes:
    <out>/aggregated/<cell>__cfgNNN.json   one per config (all-split curves + cfg)
    <out>/aggregated/<cell>__cfgNNN.csv    the val curve, for quick eyeballing
    <out>/best_<cell>__tol<mult>.json      the selected config per (cell, slack)
    <out>/best_summary.csv                 one row per (cell, slack)

Usage:
    python select_best.py --runs multirun/ --out selection/ --tols 1.0,1.1,1.25
"""
import argparse
import glob
import json
import os
import re

import pandas as pd


def _c_cols(df):
    """Constraint columns c_0, c_1, ... in numeric order."""
    return sorted([c for c in df.columns if re.fullmatch(r"c_\d+", c)],
                  key=lambda s: int(s.split("_")[1]))


def _row_maxc(df, cc, filt):
    """Per-row max constraint violation (one scalar per (seed, epoch))."""
    return df[cc].max(axis=1) if filt == "upper" else df[cc].abs().max(axis=1)


def _config_signature(meta):
    """Stable key for "the same config" across seeds (seed is NOT in here)."""
    return json.dumps(meta["hyperparameters"], sort_keys=True, default=str)


def _cell_name(cell):
    return "_".join(str(x) for x in cell)


def _decompose(df, col):
    """Two-level variance decomposition of ``col`` across the (fold, init_seed) grid.

    Returns (std_fold, std_init) Series indexed by epoch:
      std_fold = std over folds of each fold's mean (over init_seeds)  -> DATA variance
      std_init = mean over folds of each fold's std over init_seeds    -> OPTIMIZATION variance
    Both ddof=0 so a single fold / single init collapses to 0.0 (not NaN).
    """
    per_fold = df.groupby(["epoch", "fold"])[col]
    g = pd.DataFrame({"m": per_fold.mean(), "s": per_fold.std(ddof=0)}).reset_index()
    by_epoch = g.groupby("epoch")
    return by_epoch["m"].std(ddof=0), by_epoch["s"].mean()


_NON_METRIC = {"epoch", "time", "fold", "init_seed"}


def _aggregate_split(runs, split, filt):
    """Aggregate one config's per-epoch curve for one split over the (fold, init_seed)
    grid.

    Returns (DataFrame, constraint_cols) or (None, []). Metric-agnostic: EVERY numeric
    column (loss, maxc, each c_i, and the opt-split KKT metrics grad_norm/L/compl/
    lambda_i/max_viol) gets ``_mean``, ``_std`` (overall), and the variance components
    ``_std_fold`` (DATA) / ``_std_init`` (OPTIMIZATION). Non-numeric columns (e.g. the
    stringified ``acc`` array) are skipped. Also emits n_seeds / n_folds / n_inits.
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
    cols = {
        "n_seeds": g.size(),                   # total runs (folds x inits)
        "n_folds": g["fold"].nunique(),
        "n_inits": g["init_seed"].nunique(),
    }
    metric_cols = [c for c in allruns.columns
                   if c not in _NON_METRIC and pd.api.types.is_numeric_dtype(allruns[c])]
    for col in metric_cols:
        sf, si = _decompose(allruns, col)
        cols[f"{col}_mean"] = g[col].mean()
        cols[f"{col}_std"] = g[col].std(ddof=0)   # overall (across all runs)
        cols[f"{col}_std_fold"] = sf              # data variance
        cols[f"{col}_std_init"] = si              # optimization variance
    return pd.DataFrame(cols).reset_index(), cc


def _aggregate_config(runs, filt):
    """All available splits' curves for one config, aggregated over (fold, init_seed).

    Returns {'splits': {split: DataFrame}, 'm': int, 'sel_split': str} or None if
    neither val nor train is available. val is preferred (ml mode); train is used as
    the selection split when val is absent (opt/train-only mode).
    """
    splits, m = {}, 0
    for split in ("train", "val", "test", "opt"):
        res, cc = _aggregate_split(runs, split, filt)
        if res is not None:
            splits[split] = res
            m = max(m, len(cc))
    if "val" in splits:
        sel_split = "val"
    elif "train" in splits:
        sel_split = "train"
    else:
        return None
    return {"splits": splits, "m": m, "sel_split": sel_split}


def _aggregate_cell(configs, filt):
    """configs: {signature: {'meta':..., 'dirs':[...]}}.

    Returns a deterministically-ordered list of per-config aggregates:
        [{'index', 'signature', 'meta', 'splits'(dict), 'm'}, ...]
    """
    items = []
    for i, sig in enumerate(sorted(configs)):
        info = configs[sig]
        agg = _aggregate_config(info["runs"], filt)
        if agg is None:
            continue
        items.append({"index": i, "signature": sig, "meta": info["meta"], **agg})
    return items


def _write_aggregated(items, cell, agg_dir):
    """Persist each config's seed-averaged curves (all splits, per-constraint) + its
    config. Returns index->basename. The full multi-split payload lives in the JSON
    (the plotting backend reads JSON); a CSV of the selection split is written for
    eyeballing.
    """
    bases = {}
    for it in items:
        base = f"{_cell_name(cell)}__cfg{it['index']:03d}"
        bases[it["index"]] = base
        sel = it["sel_split"]
        it["splits"][sel].to_csv(os.path.join(agg_dir, base + ".csv"), index=False)
        payload = {
            "task": cell[0], "data": cell[1], "algorithm": cell[2],
            "config_index": it["index"],
            "signature": it["signature"],
            "n_seeds": int(it["splits"][sel]["n_seeds"].max()),
            "m": it["m"],
            "bound": it["meta"]["bound"],
            "select_filter": it["meta"]["select_filter"],
            "sel_split": sel,
            "hyperparameters": it["meta"]["hyperparameters"],
            "splits": {s: df.to_dict(orient="records") for s, df in it["splits"].items()},
        }
        with open(os.path.join(agg_dir, base + ".json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
    return bases


def collapse(df, tail, last_epoch):
    """Reduce one seed-mean curve to representative scalars.

    SINGLE SOURCE OF TRUTH for the curve collapse: ``plotting/aggregate_results.py``
    imports this so selection and plots use identical semantics.
    ``last_epoch`` -> mean over the last ``tail`` epochs (representative epoch = the
    window's last); else -> values at the epoch minimising the rolling-``tail`` mean
    loss. Returns a dict {epoch, loss_mean, loss_std, viol_mean, viol_std}; viol_* are
    NaN when the curve has no constraint (maxc) columns.
    """
    has_v = "maxc_mean" in df.columns

    def _get(frame, col, reduce):
        """Window-reduce a column if present, else NaN. ``reduce`` is 'mean' (window)
        or a row Series (rolling-min epoch)."""
        if col not in frame.columns if hasattr(frame, "columns") else col not in frame.index:
            return float("nan")
        return float(frame[col].mean()) if reduce == "mean" else float(frame[col])

    if last_epoch:
        t = df.tail(tail)
        epoch = int(t["epoch"].iloc[-1])
        loss_mean = float(t["loss_mean"].mean())
        loss_std = float(t["loss_std"].mean())
        loss_std_fold = _get(t, "loss_std_fold", "mean")
        loss_std_init = _get(t, "loss_std_init", "mean")
        viol_mean = float(t["maxc_mean"].mean()) if has_v else float("nan")
        viol_std = float(t["maxc_std"].mean()) if has_v else float("nan")
        viol_std_fold = _get(t, "maxc_std_fold", "mean")
        viol_std_init = _get(t, "maxc_std_init", "mean")
    else:
        smooth = df["loss_mean"].rolling(tail, min_periods=1).mean()
        row = df.loc[smooth.idxmin()]
        epoch = int(row["epoch"])
        loss_mean, loss_std = float(row["loss_mean"]), float(row["loss_std"])
        loss_std_fold = _get(row, "loss_std_fold", "row")
        loss_std_init = _get(row, "loss_std_init", "row")
        viol_mean = float(row["maxc_mean"]) if has_v else float("nan")
        viol_std = float(row["maxc_std"]) if has_v else float("nan")
        viol_std_fold = _get(row, "maxc_std_fold", "row")
        viol_std_init = _get(row, "maxc_std_init", "row")
    return {"epoch": epoch, "loss_mean": loss_mean, "loss_std": loss_std,
            "loss_std_fold": loss_std_fold, "loss_std_init": loss_std_init,
            "viol_mean": viol_mean, "viol_std": viol_std,
            "viol_std_fold": viol_std_fold, "viol_std_init": viol_std_init}


def _stats_at(df, epoch, prefix):
    """Report a split's stats at a fixed epoch (the val-selected epoch), prefixed.
    Used to read the held-out TEST numbers at the same epoch the val selection picked."""
    sub = df[df["epoch"] == epoch]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    out = {f"{prefix}_loss_mean": float(r["loss_mean"]), f"{prefix}_loss_std": float(r["loss_std"])}
    for k in ("loss_std_fold", "loss_std_init"):
        if k in sub.columns:
            out[f"{prefix}_{k}"] = float(r[k])
    if "maxc_mean" in sub.columns:
        out[f"{prefix}_maxc_mean"] = float(r["maxc_mean"])
        out[f"{prefix}_maxc_std"] = float(r["maxc_std"])
        for k in ("maxc_std_fold", "maxc_std_init"):
            if k in sub.columns:
                out[f"{prefix}_{k}"] = float(r[k])
    return out


def _select(items, filt, tol, tail, last_epoch):
    """Windowed-collapse each config's selection-split curve, then pick the feasible
    config with the minimum representative loss. Selection split is val (ml mode) or
    train (opt mode, when val is absent)."""
    sel = items[0].get("sel_split", "val")
    has_maxc = "maxc_mean" in items[0]["splits"][sel].columns
    rows = []
    for it in items:
        c = collapse(it["splits"][sel], tail, last_epoch)
        rows.append({"config_index": it["index"],
                     "n_seeds": int(it["splits"][sel]["n_seeds"].max()), **c})
    pool = pd.DataFrame(rows)
    feasible = pool if (filt == "none" or not has_maxc) else pool[pool["viol_mean"] <= tol]
    if feasible.empty:
        return None
    row = feasible.loc[feasible["loss_mean"].idxmin()]
    return {
        "config_index": int(row["config_index"]),
        "best_epoch": int(row["epoch"]),
        "n_seeds": int(row["n_seeds"]),
        "sel_split": sel,
        "val_loss_mean": float(row["loss_mean"]),
        "val_loss_std": float(row["loss_std"]),
        "val_loss_std_fold": float(row["loss_std_fold"]),   # data variance
        "val_loss_std_init": float(row["loss_std_init"]),   # optimization variance
        "val_maxc_mean": float(row["viol_mean"]) if has_maxc else float("nan"),
        "val_maxc_std": float(row["viol_std"]) if has_maxc else float("nan"),
        "val_maxc_std_fold": float(row["viol_std_fold"]) if has_maxc else float("nan"),
        "val_maxc_std_init": float(row["viol_std_init"]) if has_maxc else float("nan"),
    }


def _parse_tols(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    
    ## READ CLI ##
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="multirun root to scan recursively")
    ap.add_argument("--out", default="selection", help="output directory")
    ap.add_argument(
        "--tols", default="1.0,1.1,1.25",
        help="comma-separated feasibility-slack multipliers; one feasibility-first "
             "selection per multiplier (tol = bound * mult). Ignored for "
             "select_filter='none' (e.g. adam), which yields a single pick.",
    )
    ap.add_argument("--tail", type=int, default=5,
                    help="window length for the per-config val collapse (mean of the "
                         "last `tail` epochs, or rolling window with --rolling)")
    ap.add_argument("--rolling", action="store_true",
                    help="select the epoch minimising the rolling-`tail` mean val loss "
                         "(default: mean over the last `tail` epochs)")
    ap.add_argument("--approach", default=None, choices=["ml", "opt"],
                    help="only aggregate runs of this approach (ml: CV/val-selected; "
                         "opt: full-data, train-selected, with KKT metrics). Required if "
                         "the scanned tree mixes approaches -- ml and opt must not share "
                         "a cell. Use a separate --out per approach (e.g. selection/opt).")
    args = ap.parse_args()
    tol_mults = _parse_tols(args.tols)
    last_epoch = not args.rolling
    agg_dir = os.path.join(args.out, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)

    # Discover runs; nest as cell -> config signature -> {meta, runs:[{dir,fold,init_seed}]}.
    # ml and opt runs differ in splits/metrics, so they must not merge into one cell:
    # filter by --approach (or require it if the tree mixes approaches).
    cells = {}
    approaches_seen = set()
    for meta_path in glob.glob(os.path.join(args.runs, "**", "run_meta.json"), recursive=True):
        d = os.path.dirname(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        appr = meta.get("approach", "ml")
        if args.approach is not None and appr != args.approach:
            continue
        approaches_seen.add(appr)
        cell = (meta["task"], meta.get("data"), meta["algorithm"])
        sig = _config_signature(meta)
        entry = cells.setdefault(cell, {}).setdefault(sig, {"meta": meta, "runs": []})
        entry["runs"].append({"dir": d, "fold": int(meta.get("fold", 0)),
                              "init_seed": int(meta.get("init_seed", 0))})

    if args.approach is None and len(approaches_seen) > 1:
        print(f"Runs under {args.runs} mix approaches {sorted(approaches_seen)}; ml and "
              f"opt cannot share a cell. Re-run with --approach ml or --approach opt "
              f"(and a per-approach --out, e.g. --out selection/opt).")
        return

    if not cells:
        print(f"No runs (run_meta.json) found under {args.runs}")
        return

    summary = []
    for cell, configs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        any_meta = next(iter(configs.values()))["meta"]
        filt = any_meta["select_filter"]
        bound = any_meta["bound"]

        items = _aggregate_cell(configs, filt)
        if not items:
            print(f"[skip] {_cell_name(cell)}: no train or val csv")
            continue
        bases = _write_aggregated(items, cell, agg_dir)

        # Threshold sweep: one feasibility-first selection per slack multiplier.
        # filter='none' (e.g. adam) has no feasibility notion -> a single pick.
        mults = [None] if filt == "none" else tol_mults
        for mult in mults:
            tol = None if filt == "none" else bound * mult
            tag = "none" if mult is None else f"{mult:g}"
            best = _select(items, filt, tol, args.tail, last_epoch)
            if best is None:
                print(f"[infeasible] {_cell_name(cell)} tol_mult={tag}: "
                      f"no config met tol={tol} (on seed-mean)")
                continue

            base = bases[best["config_index"]]
            winner = next(it for it in items if it["index"] == best["config_index"])
            # Held-out TEST numbers for the winner, at the val-selected epoch.
            test_stats = (_stats_at(winner["splits"]["test"], best["best_epoch"], "test")
                          if "test" in winner["splits"] else {})
            record = {
                "task": cell[0], "data": cell[1], "algorithm": cell[2],
                "filter": filt, "tol_mult": mult, "tol": tol, "n_configs": len(items),
                "tail": args.tail, "select_mode": "rolling" if args.rolling else "last_mean",
                **best,
                **test_stats,
                "aggregated_file": base + ".json",   # link back to the saved curves
                "best_hyperparameters": winner["meta"]["hyperparameters"],
            }
            suffix = "" if mult is None else f"__tol{tag}"
            with open(os.path.join(args.out, f"best_{_cell_name(cell)}{suffix}.json"), "w") as f:
                json.dump(record, f, indent=2, default=str)
            summary.append({k: v for k, v in record.items() if k != "best_hyperparameters"})
            tline = (f" | test {test_stats['test_loss_mean']:.5f}±{test_stats['test_loss_std']:.5f}"
                     if "test_loss_mean" in test_stats else "")
            print(f"[best] {_cell_name(cell)} tol_mult={tag}: "
                  f"{best.get('sel_split','val')} loss={best['val_loss_mean']:.5f} "
                  f"(±{best['val_loss_std_init']:.4f} init, ±{best['val_loss_std_fold']:.4f} fold) "
                  f"max_c={best['val_maxc_mean']:.5f}{tline} "
                  f"(cfg{best['config_index']:03d}, epoch {best['best_epoch']}, "
                  f"{best['n_seeds']} runs, {len(items)} configs)")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(args.out, "best_summary.csv"), index=False)
        print(f"\nWrote {len(summary)} rows; per-config aggregates in {agg_dir}/")


if __name__ == "__main__":
    main()
