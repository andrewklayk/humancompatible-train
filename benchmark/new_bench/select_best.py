"""Select the best config per cell from aggregated curves.

Stage 2 of 2 (stage 1 = aggregate.py). Reads ``--agg``/*.json (per-config
seed-averaged curves) -- NEVER the raw runs -- groups them into (task, data,
algorithm) cells, and per cell collapses each config's selection-split curve (val,
or train when there is no val) to a representative (loss, violation) via ``collapse``
(shared with plotting/aggregate_results.py so selection and plots agree). Among
configs feasible at ``bound * mult`` it takes min loss; one pick per ``--tols``
multiplier (select_filter='none', e.g. adam -> plain argmin loss).

Writes under ``--out``:
    best_<cell>__tol<mult>.json   selected config per (cell, feasibility slack)
    best_summary.csv              one row per (cell, slack)

Usage:
    python aggregate.py    --runs multirun/ --out selection/ --approach ml
    python select_best.py  --agg selection/aggregated --out selection/ --tols 1.0,1.1,1.25
"""
import argparse
import glob
import json
import os

import pandas as pd


def _cell_name(cell):
    return "_".join(str(x) for x in cell)


def collapse(df, tail, last_epoch):
    """Reduce a seed-mean curve to representative scalars (shared with plotting).

    last_epoch=True -> mean over the last ``tail`` epochs (epoch = window end);
    else -> the row at the epoch minimising the rolling-``tail`` mean loss. Returns
    {epoch, loss_mean/std/std_fold/std_init, viol_mean/std/std_fold/std_init}; viol_*
    map to the max_viol columns and are NaN when the curve has no constraints.
    """
    if last_epoch:
        win = df.tail(tail)
        r = win.mean(numeric_only=True)
        r["epoch"] = win["epoch"].iloc[-1]
    else:
        r = df.loc[df["loss_mean"].rolling(tail, min_periods=1).mean().idxmin()]

    def v(col):
        return float(r[col]) if col in r.index else float("nan")

    return {"epoch": int(r["epoch"]),
            "loss_mean": v("loss_mean"), "loss_std": v("loss_std"),
            "loss_std_fold": v("loss_std_fold"), "loss_std_init": v("loss_std_init"),
            "viol_mean": v("max_viol_mean"), "viol_std": v("max_viol_std"),
            "viol_std_fold": v("max_viol_std_fold"), "viol_std_init": v("max_viol_std_init")}


def _prefix_collapse(c, prefix):
    """Rename a collapse dict's loss_*/viol_* to <prefix>_loss_*/<prefix>_max_viol_*."""
    out = {f"{prefix}_loss_{k}": c[f"loss_{k}"] for k in ("mean", "std", "std_fold", "std_init")}
    out.update({f"{prefix}_max_viol_{k}": c[f"viol_{k}"] for k in ("mean", "std", "std_fold", "std_init")})
    return out


def _stats_at(df, epoch, prefix):
    """Prefixed loss/max_viol stats of one split at a fixed epoch (the selected epoch)."""
    sub = df[df["epoch"] == epoch]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    keys = ("loss_mean", "loss_std", "loss_std_fold", "loss_std_init",
            "max_viol_mean", "max_viol_std", "max_viol_std_fold", "max_viol_std_init")
    return {f"{prefix}_{k}": float(r[k]) for k in keys if k in sub.columns}


def _select(items, filt, tol, tail, last_epoch, split=None):
    """Collapse each config's selection-split curve; return the feasible min-loss
    winner's collapse dict (+ config_index, n_seeds), or None if none is feasible."""
    sel = items[0]["sel_split"] if split is None else split
    rows = [{"config_index": it["index"],
             "n_seeds": int(it["splits"][sel]["n_seeds"].max()),
             **collapse(it["splits"][sel], tail, last_epoch)} for it in items]
    pool = pd.DataFrame(rows)
    filtered = filt != "none" and pool["viol_mean"].notna().any()
    feasible = pool[pool["viol_mean"] <= tol] if filtered else pool
    if feasible.empty:
        return None
    return feasible.loc[feasible["loss_mean"].idxmin()].to_dict()


def _load_cells(agg_dir):
    """Read aggregate.py's per-cell CSV+JSON into {(task, data, algo): [item, ...]}.

    Each ``<cell>.json`` holds per-config metadata; the sibling ``<cell>.csv`` holds
    all curves (long: one row per config/split/epoch). Each item carries its split
    curves as DataFrames plus the fields selection needs.
    """
    cells = {}
    for jpath in sorted(glob.glob(os.path.join(agg_dir, "*.json"))):
        with open(jpath) as f:
            meta = json.load(f)
        long = pd.read_csv(jpath[:-5] + ".csv")
        items = []
        for cfg in meta["configs"]:
            idx = int(cfg["config_index"])
            sub = long[long["config"] == idx]
            items.append({
                "index": idx,
                "sel_split": cfg["sel_split"],
                "filter": meta["select_filter"],
                "bound": meta["bound"],
                "hyperparameters": cfg["hyperparameters"],
                "agg_file": os.path.basename(jpath),
                "splits": {s: g.drop(columns=["config", "split"]).sort_values("epoch").reset_index(drop=True)
                           for s, g in sub.groupby("split")},
            })
        cells[(meta["task"], meta.get("data"), meta["algorithm"])] = items
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="selection/aggregated",
                    help="dir of aggregate.py's per-config JSONs (run aggregate.py first)")
    ap.add_argument("--out", default="selection", help="output directory for best_*.json")
    ap.add_argument("--tols", default="1.0,1.1,1.25",
                    help="comma-separated feasibility-slack multipliers (tol = bound * mult); "
                         "one pick each. Ignored for select_filter='none' (adam).")
    ap.add_argument("--tail", type=int, default=5,
                    help="window for the per-config collapse (mean of the last `tail` epochs)")
    ap.add_argument("--rolling", action="store_true",
                    help="instead select the epoch minimising the rolling-`tail` mean loss")
    ap.add_argument("--selection_split", default="opt",
                    help="which split to select by; defaults to opt.")
    args = ap.parse_args()
    tol_mults = [float(x) for x in args.tols.split(",") if x.strip()]
    last_epoch = not args.rolling
    os.makedirs(args.out, exist_ok=True)

    cells = _load_cells(args.agg)
    if not cells:
        print(f"No aggregated configs (*.json) found under {args.agg}; run aggregate.py first.")
        return

    summary = []
    for cell, items in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        items.sort(key=lambda it: it["index"])
        by_index = {it["index"]: it for it in items}
        filt, bound = items[0]["filter"], items[0]["bound"]

        # One feasibility-first pick per slack multiplier (filter='none' -> single pick).
        for mult in ([None] if filt == "none" else tol_mults):
            tol = None if mult is None else bound * mult
            tag = "none" if mult is None else f"{mult:g}"
            best = _select(items, filt, tol, args.tail, last_epoch, args.selection_split)
            if best is None:
                print(f"[infeasible] {_cell_name(cell)} tol_mult={tag}: none met tol={tol}")
                continue

            epoch = int(best["epoch"])
            winner = by_index[int(best["config_index"])]
            test_stats = (_stats_at(winner["splits"]["test"], epoch, "test")
                          if "test" in winner["splits"] else {})
            record = {
                "task": cell[0], "data": cell[1], "algorithm": cell[2],
                "filter": filt, "tol_mult": mult, "tol": tol, "n_configs": len(items),
                "tail": args.tail, "select_mode": "rolling" if args.rolling else "last_mean",
                "config_index": int(best["config_index"]), "best_epoch": epoch,
                "n_seeds": int(best["n_seeds"]), "sel_split": winner["sel_split"],
                **_prefix_collapse(best, args.selection_split), **test_stats,
                "aggregated_file": winner["agg_file"],
                "best_hyperparameters": winner["hyperparameters"],
            }
            suffix = "" if mult is None else f"__tol{tag}"
            with open(os.path.join(args.out, f"best_{_cell_name(cell)}{suffix}.json"), "w") as f:
                json.dump(record, f, indent=2, default=str)
            summary.append({k: v for k, v in record.items() if k != "best_hyperparameters"})

            tline = (f" | test {test_stats['test_loss_mean']:.5f}±{test_stats['test_loss_std']:.5f}"
                     if "test_loss_mean" in test_stats else "")
            print(f"[best] {_cell_name(cell)} tol_mult={tag}: {winner['sel_split']} "
                  f"loss={best['loss_mean']:.5f} (±{best['loss_std_init']:.4f} init, "
                  f"±{best['loss_std_fold']:.4f} fold) max_c={best['viol_mean']:.5f}{tline} "
                  f"(cfg{int(best['config_index']):03d}, epoch {epoch}, "
                  f"{int(best['n_seeds'])} runs, {len(items)} configs)")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(args.out, "best_summary.csv"), index=False)
        print(f"\nWrote {len(summary)} best-config picks to {args.out}/")


if __name__ == "__main__":
    main()
