"""
plot_style.py — NeurIPS-style matplotlib defaults + shared method styling.

NeurIPS text width is ~5.5in single-column-ish (the body is ~5.5in wide for a
single column in the 2-col-ish layout used by the checklist template). Figures
are usually placed at column width (~3.25in) or text width (~5.5in). Fonts at
8pt so they read at print size. Import `set_neurips_style()` once at the top of
a plotting script.
"""

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# Consistent per-method color + label across all figures.
METHOD_STYLE = {
    "adam":      {"label": "Adam (unconstr.)", "color": "#4C4C4C", "ls": "--"},
    "pbm":       {"label": "SPBM (ours)",      "color": "#D1495B", "ls": "-"},
    "alm_proj": {"label": "SSL-ALM (proj)",  "color": "#2E86AB", "ls": "-"},
    "alm_max":   {"label": "SSL-ALM (max)",    "color": "#5BC0BE", "ls": "-"},
    "ssg":       {"label": "SSw",              "color": "#E0A458", "ls": "-"},
}

# NeurIPS column / text widths in inches.
COL_WIDTH = 3.25
TEXT_WIDTH = 5.50


def set_neurips_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def style_for(method):
    return METHOD_STYLE.get(
        method, {"label": method, "color": "#888888", "ls": "-"}
    )