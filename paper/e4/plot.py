"""
Aggregates the runs in the results/runs folder and plots the best possible run for each
method - best possible is selected based on the best train loss and the constraint satisfaction
from all feasible plot the one with the smallest loss function in the last iteration

    python plot.py                  # x axis = optimizer updates
    python plot.py --x time --tol 1e-3
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper import _harness

STYLE_KEY = {"unconstrained": "adam", "spbm": "pbm", "alm": "alm_proj"}   # e4 method -> plot_style key
PANELS = {"loss": "train loss", "val_loss": "val loss",
          "constraint_max_viol": "max constraint violation", "network_size": "network size [%]"}
TICK_FORMAT = {"loss": "%.2f", "val_loss": "%.2f", "network_size": "%.1f"}


def load_runs(root):
    """Every row of every run, tagged with its method, config hash and seed."""
    frames = []
    for csv_path in sorted(Path(root).glob("*/metrics.csv")):
        params = json.loads((csv_path.parent / "params.json").read_text())
        cfg = csv_path.parent.name.rsplit("_", 1)[1]   # run dirs end in the seed-independent config hash
        frames.append(pd.read_csv(csv_path).assign(method=params["method"], cfg=cfg, seed=params["seed"]))
    if not frames:
        sys.exit(f"no runs found in {root}")
    return pd.concat(frames, ignore_index=True)


def pick_best(runs, tol):
    """Per method: among configs feasible at the last row (seed mean), the one with the smallest train loss."""
    last = runs.sort_values("update").groupby(["method", "cfg", "seed"]).tail(1)
    final = last.groupby(["method", "cfg"]).agg(loss=("loss", "mean"), viol=("constraint_max_viol", "mean"),
                                                 seeds=("seed", "nunique")).reset_index()
    final["feasible"] = (final["method"] == "unconstrained") | (final["viol"] <= tol)
    best = []
    for method, g in final.groupby("method"):
        feasible = g[g["feasible"]]
        # shortcut: a method with no feasible config shows its least-violating one, flagged in the legend
        best.append(feasible.loc[feasible["loss"].idxmin()] if len(feasible) else g.loc[g["viol"].idxmin()])
    return pd.DataFrame(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--x", choices=["update", "time"], default="update")
    parser.add_argument("--tol", type=float, default=0.01, help="feasible if max constraint <= tol")
    args = parser.parse_args()

    runs = load_runs(args.runs)

    best = pick_best(runs, args.tol)
    print(best.to_string(index=False))

    fig, axes, plt = _harness.figure(2, 2)
    for _, b in best.iterrows():
        style = _harness._load_plot_style().style_for(STYLE_KEY.get(b["method"], b["method"]))
        label = style["label"] + ("" if b["feasible"] else " (infeasible)")
        # mean +- std over seeds, aligned on the update index (wall time differs per seed)
        stats = runs[(runs["method"] == b["method"]) & (runs["cfg"] == b["cfg"])].groupby("update")[
            ["time_s", *PANELS]].agg(["mean", "std"])
        
        x = stats[("time_s", "mean")] / 60 if args.x == "time" else stats.index
        for ax, (col, name) in zip(axes, PANELS.items()):
            if col == "constraint_max_viol" and b["method"] == "unconstrained":
                continue   # unconstrained logs a placeholder, not a real constraint
            mean, std = stats[(col, "mean")], stats[(col, "std")]
            ax.plot(x, mean, color=style["color"], ls=style["ls"], label=label if ax is axes[0] else None)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.2, lw=0)
            ax.set_ylabel(name)
            if col in TICK_FORMAT:
                ax.yaxis.set_major_formatter(plt.FormatStrFormatter(TICK_FORMAT[col]))
            else:
                ax.ticklabel_format(axis="y", useOffset=False)   # show 0.23, not an offset like "+2.3e-1"

    axes[2].axhline(args.tol, color="0.6", lw=0.5, ls=(0, (1, 2)))   # feasibility threshold
    for ax in axes[2:]:
        ax.set_xlabel("wall time [min]" if args.x == "time" else "optimizer updates")
    _harness._load_plot_style().top_legend(fig, axes[0])
    _harness.save_figure(fig, f"best_runs_{args.x}", "e4")


if __name__ == "__main__":
    main()
