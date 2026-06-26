"""
plot_pareto.py — Loss vs. constraint-violation scatter.

Each point = one hyperparameter configuration of one algorithm.
    x  = max constraint violation (across constraint dims), mean ± std over seeds
    y  = train loss, mean ± std over seeds

Different algorithms use different colours and markers.
A vertical dotted line marks the feasibility bound.

Usage:
    edit SPEC at the bottom, then:  python3 plot_pareto.py
"""

import numpy as np

from plot_style import set_neurips_style, style_for, COL_WIDTH
from aggregate_results import ExperimentSpec, aggregate_experiment
import matplotlib.pyplot as plt


def _pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (minimise both x and y)."""
    mask = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i != j and xs[j] <= xs[i] and ys[j] <= ys[i] and (xs[j] < xs[i] or ys[j] < ys[i]):
                mask[i] = False
                break
    return mask


METHOD_MARKER = {
    "adam":     "o",
    "pbm":      "s",
    "alm_proj": "^",
    "alm_max":  "D",
    "ssg":      "v",
}
_FALLBACK_MARKERS = ["P", "X", "h", "*", "p"]


def plot_scatter(
    spec: ExperimentSpec,
    methods=None,
    out="pareto.pdf",
    tail=10,
    split="train",
    viol_stat="mean",
    pareto_filter=False,
):
    """Scatter each hyperparameter config as (violation, loss) with seed error bars.

    Parameters
    ----------
    viol_stat     : "mean" — violation_constr_mean ± violation_constr_std on the x-axis.
                    "max"  — violation_constr_max on the x-axis (worst seed; no x error bar).
    pareto_filter : if True, show only non-dominated configs per method (minimise both axes).
    """
    set_neurips_style()
    agg = aggregate_experiment(spec, tail=tail, split=split)
    if methods is None:
        methods = [m for m in ["pbm", "alm_proj", "alm_max", "ssg"]
                   if m in agg]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.85))

    _fallback_idx = 0
    for m in methods:
        df = agg[m]
        st = style_for(m)
        marker = METHOD_MARKER.get(m, _FALLBACK_MARKERS[_fallback_idx % len(_FALLBACK_MARKERS)])
        if m not in METHOD_MARKER:
            _fallback_idx += 1

        ys    = df["loss_mean"].values
        y_err = df["loss_std"].values

        xs    = df["violation_constr_mean"].values - spec.bound
        x_err = df["violation_constr_std"].values

        if pareto_filter:
            mask  = _pareto_mask(xs, ys)
            xs    = xs[mask]
            ys    = ys[mask]
            y_err = y_err[mask]
            if x_err is not None:
                x_err = x_err[mask]

        ax.errorbar(
            xs, ys,
            xerr=x_err, yerr=y_err,
            fmt=marker, color=st["color"], label=st["label"],
            ms=4, lw=0.5, capsize=2, capthick=0.5,
            zorder=3, alpha=0.85,
        )

    ax.axvline(0, color="#888888", lw=0.8, ls=":", zorder=1)

    x_label = ("Max constraint violation across seeds"
               if viol_stat == "max"
               else "Mean max constraint violation (± std over seeds)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Train loss (mean ± std over seeds)")
    # ax.legend(loc="upper right")
    fig.legend(loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])   # leave room for top legend
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS = os.path.join(REPO_ROOT, "results")

    name = "E3"
    spec = ExperimentSpec(
        name=name,
        data="folktables",
        task="folktables_positive_rate_pair",
        bound=0.1,
        seeds=(0, 1, 2, 3, 4),
        results_root=RESULTS,
    )

    plot_scatter(spec, out=os.path.join(REPO_ROOT, f"./results/plots/pareto_{name}.pdf"),
                 tail=1, split="train", pareto_filter=True)
