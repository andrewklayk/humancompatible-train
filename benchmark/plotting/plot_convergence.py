"""
plot_convergence.py — Anytime / convergence figure.

For each method, takes the VALIDATION-selected best config (from best_{method}.json
written by the gridsearch) and plots that config's trajectory over epochs, averaged
across seeds with a +/-1 std band. Two stacked panels:
    (top)    loss vs epoch
    (bottom) max constraint vs epoch, with the threshold line

HPs are selected on VALIDATION (leakage-free); we DISPLAY the train trajectory
(optimizer dynamics) by default, switchable to test. Selecting on val + showing
train is standard and is what the SPBM paper's own figures do.

Usage:
    edit SPEC at the bottom, then:  python3 plot_convergence.py
"""

import os
import re
import numpy as np
import pandas as pd

from plot_style import set_neurips_style, style_for, COL_WIDTH
from aggregate_results import ExperimentSpec, load_best_config
import matplotlib.pyplot as plt


def _trajectory_for_config(spec, method, config_idx, split):
    """Return (epochs, loss_mean, loss_std, viol_mean, viol_std) for one config,
    averaged across seeds. None if no files."""
    per_seed_loss, per_seed_viol, epochs_ref = [], [], None
    for seed in spec.seeds:
        path = os.path.join(spec.seed_dir(seed), f"runs_{method}_{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        first = df.columns[0]
        if first.startswith("Unnamed") or first == "":
            df = df.rename(columns={first: "config"})
        sub = df[df["config"] == config_idx].sort_values("epoch")
        if sub.empty:
            continue
        c_cols = [c for c in df.columns if re.fullmatch(r"c_\d+", c)]
        epochs = sub["epoch"].values
        loss = sub["loss"].values
        viol = sub[c_cols].max(axis=1).values  # one-sided max constraint
        if epochs_ref is None:
            epochs_ref = epochs
        per_seed_loss.append(loss)
        per_seed_viol.append(viol)

    if not per_seed_loss:
        return None
    # align lengths (defensive: seeds should share epoch count)
    L = min(len(x) for x in per_seed_loss)
    loss_arr = np.stack([x[:L] for x in per_seed_loss])
    viol_arr = np.stack([x[:L] for x in per_seed_viol])
    return (epochs_ref[:L],
            loss_arr.mean(0), loss_arr.std(0),
            viol_arr.mean(0), viol_arr.std(0))


def plot_convergence(spec, methods=None, display_split="train", out="conv.pdf"):
    set_neurips_style()
    if methods is None:
        methods = ["adam", "pbm", "alm_proj", "alm_max", "ssg"]

    fig, (ax_loss, ax_c) = plt.subplots(
        2, 1, figsize=(COL_WIDTH, COL_WIDTH * 1.4), sharex=True
    )

    plotted = []
    for m in methods:
        # pick the val-selected best config; need one config index for the curve.
        # use seed 0's best_{method}.json (val-selected); if absent, skip.
        best = load_best_config(spec, m, spec.seeds[0])
        if best is None:
            print(f"  {m}: no best_{m}.json (seed {spec.seeds[0]}), skipping")
            continue
        cfg = best.get("best_idx", best.get("best_index"))
        if cfg is None:
            print(f"  {m}: best json has no config index, skipping")
            continue
        traj = _trajectory_for_config(spec, m, cfg, display_split)
        if traj is None:
            print(f"  {m}: no trajectory for config {cfg}, skipping")
            continue
        ep, lm, ls_, vm, vs = traj
        st = style_for(m)
        ax_loss.plot(ep, lm, color=st["color"], ls=st["ls"], label=st["label"])
        ax_loss.fill_between(ep, lm - ls_, lm + ls_, color=st["color"], alpha=0.15, lw=0)
        ax_c.plot(ep, vm, color=st["color"], ls=st["ls"])
        ax_c.fill_between(ep, vm - vs, vm + vs, color=st["color"], alpha=0.15, lw=0)
        plotted.append(m)

    # constraint threshold
    ax_c.axhline(spec.bound, color="red", ls=":", lw=0.8, label="threshold")

    ax_loss.set_ylabel(f"{display_split.capitalize()} loss")
    ax_c.set_ylabel("Max constraint")
    ax_c.set_xlabel("Epoch")
    ax_loss.legend(loc="upper right", ncol=1)
    fig.align_ylabels([ax_loss, ax_c])
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}  (methods: {', '.join(plotted)})")
    return out


if __name__ == "__main__":
    import os
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS = os.path.join(REPO_ROOT, "results")

    name = 'E3'
    spec = ExperimentSpec(
        name=name,
        data="folktables",                       # <-- match your real results/ dir prefix
        task="folktables_positive_rate_pair",    # <-- match your real task name
        bound=0.1,
        pinns=False,
        seeds=(0, 1, 2),
        results_root=RESULTS,
    )
    # display_split="train" shows optimizer dynamics; switch to "val" or "test"
    plot_convergence(spec, display_split="train",
                     out=os.path.join(REPO_ROOT, f"./results/plots/convergence_{name}.pdf"))