"""
plot_cdf.py — Robustness CDF / performance profile.

For each method, plots the fraction of ALL its grid configs that are BOTH
feasible (mean constraint <= bound) AND have train loss <= L, as a function of L.

Reading the figure:
  - Higher curve  = more configs reach that (loss, feasible) bar = more robust.
  - Left-shifted  = reaches low loss with fewer wasted configs.
  - Plateau height = the method's overall feasibility rate (fraction of the grid
    that is feasible at all). A curve can never rise above its feasibility rate.

Infeasible configs are NOT placed at +inf on the axis; they simply never enter
the numerator, so each curve plateaus at its feasibility rate. The x-axis stays
in real loss units.

Feasibility is strict on the seed-MEAN violation (<= bound). The seed averaging
already absorbs per-seed mini-batch noise, so no extra tolerance is applied; the
printed diagnostic reports viol_std so you can confirm the noise is small.

Usage:
    edit SPEC at the bottom, then:  python3 plot_cdf.py
"""

import numpy as np
import pandas as pd

from plot_style import set_neurips_style, style_for, COL_WIDTH
from aggregate_results import ExperimentSpec, aggregate_experiment
import matplotlib.pyplot as plt


def _print_frontier(agg: dict, bound: float):
    """Diagnostic: per method, the feasible configs sorted by loss (the frontier
    the CDF draws). Printed so you SEE what the curve will show before trusting it."""
    print("\n--- feasible frontier (sorted by loss) ---")
    for method, df in agg.items():
        feas = df[df["feasible_mean_viol"]].sort_values("loss_mean")
        n_feas, n_tot = len(feas), len(df)
        best = f"{feas['loss_mean'].iloc[0]:.3f}" if n_feas else "—"
        print(f"  {method:10s}: {n_feas:3d}/{n_tot} feasible  "
              f"| best feasible loss {best}  "
              f"| median viol_std {df['viol_std'].median():.4f}")
        if n_feas:
            head = feas[["config", "loss_mean", "viol_mean"]].head(5)
            for _, r in head.iterrows():
                print(f"        cfg {int(r['config']):>3}  "
                      f"loss {r['loss_mean']:.3f}  viol {r['viol_mean']:.3f}")


def _cdf_curve(df: pd.DataFrame, loss_grid: np.ndarray) -> np.ndarray:
    """fraction of ALL configs that are feasible AND loss_mean <= L, for each L."""
    n_total = len(df)
    feas = df[df["feasible_mean_viol"]]
    losses = np.sort(feas["loss_mean"].values)
    # count of feasible configs with loss <= L, divided by total config count
    counts = np.searchsorted(losses, loss_grid, side="right")
    return counts / n_total


def plot_cdf(spec: ExperimentSpec, methods=None, out="cdf.pdf"):
    set_neurips_style()
    agg = aggregate_experiment(spec)
    if methods is None:
        methods = [m for m in ["adam", "pbm", "alm_proj", "alm_max", "ssg"]
                   if m in agg]

    _print_frontier(agg, spec.bound)

    # shared loss grid spanning all methods' feasible losses
    all_feas_losses = np.concatenate([
        agg[m][agg[m]["feasible_mean_viol"]]["loss_mean"].values
        for m in methods if (agg[m]["feasible_mean_viol"]).any()
    ]) if any((agg[m]["feasible_mean_viol"]).any() for m in methods) else np.array([0.0, 1.0])
    lo, hi = all_feas_losses.min(), all_feas_losses.max()

    hi = 0.9

    pad = 0.05 * (hi - lo + 1e-9)
    loss_grid = np.linspace(lo - pad, hi + pad, 400)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.8))
    for m in methods:
        st = style_for(m)
        y = _cdf_curve(agg[m], loss_grid)
        ax.plot(loss_grid, y, label=st["label"], color=st["color"],
                ls=st["ls"], drawstyle="steps-post")

    ax.set_xlabel("Train loss threshold $L$")
    ax.set_ylabel("Frac. configs feasible \\& $\\leq L$")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    import os
    # results/ lives at the repo root (one level up from this plotting/ folder),
    # resolved absolutely so it works whether you run from the repo root
    # (`python3 plotting/plot_cdf.py`) or from inside plotting/.
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS = os.path.join(REPO_ROOT, "results")

    name = 'E3'
    spec = ExperimentSpec(
        name=name,
        data="folktables",                       # <-- match your real results/ dir prefix
        task="folktables_positive_rate_pair",    # <-- match your real task name
        bound=0.1,
        seeds=(0, 1, 2),
        results_root=RESULTS,
    )

    # write the figure next to results/, at the repo root
    plot_cdf(spec, out=os.path.join(REPO_ROOT, f"./results/plots/cdf_{name}.pdf"))