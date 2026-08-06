"""plot_style.py — single style source for all paper figures."""


import matplotlib
matplotlib.use("Agg")  # headless
import numpy as np
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.9          # inches, full 2-column width
COL_WIDTH = TEXT_WIDTH / 2
ROW_H = 2.0

METHOD_COLORS = {
    "SPBM": "#D1495B",            # crimson — your method
    "Adam": "#4C4C4C",            # muted baseline
    "SSL-ALM (proj.)": "#2E86AB",
    "SSL-ALM (max)": "#5BC0BE",
    "SSw": "#E0A458",
}
METHOD_LABELS = {
    "adam": "Adam", "pbm": "SPBM",
    "alm_proj": "SSL-ALM (proj.)", # make it work in maths
    "alm_max": "SSL-ALM (max)", "ssg": "SSw",
}
_MARKERS = {"adam": "o", "pbm": "s",
            "alm_proj": "D", "alm_max": "^", "ssg": "v"}
_LS = {"adam": "--", "pbm": "-", "alm_proj": "-",  "alm_max": "-", "ssg": ":"}


def style_for(method):
    """Accepts method key ('pbm') or display label ('SPBM')."""
    label = METHOD_LABELS.get(method, method)
    key = method if method in METHOD_LABELS else \
          next((k for k, v in METHOD_LABELS.items() if v == method), method)
    return {"label": label, "color": METHOD_COLORS.get(label, "#888888"),
            "ls": _LS.get(key, "-"), "marker": _MARKERS.get(key, "o")}


def set_neurips_style():
    plt.rcParams.update({
        "font.family": "serif",          # serif body (matches paper)
        "mathtext.fontset": "cm",        # Computer-Modern math (fixes "weird math")
        "axes.spines.top": False,        # drop top spine
        "axes.spines.right": False,      # drop right spine (removes the box)
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.0, "axes.linewidth": 0.6,
        "grid.linestyle": "--", "grid.alpha": 0.4, "axes.grid": True,
        "legend.frameon": False, "savefig.bbox": "tight",
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def top_legend(fig, ax):
    """Your renderer's shared top legend, as a reusable helper."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])