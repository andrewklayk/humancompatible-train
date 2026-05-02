
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────────────
THRESHOLD = 0.1 * 1.1
EXPERIMENTS = ['cifar10_val', 'cifar10_train']
EXPERIMENTS_NAMES = ['CIFAR10', 'CIFAR10_TRAIN']
# METHODS = ['pbm_dimin']
METHODS = ['pbm_dimin', 'sslalm']

def load_last_epoch(csv_path):
    df = pd.read_csv(csv_path)
    last_epoch = df['epoch'].max()
    df = df[df['epoch'] == last_epoch].copy()
    constraint_cols = [col for col in df.columns if col.startswith('c_')]
    hp_cols = [col for col in df.columns if '__' in col]
    df['max_constraint'] = df[constraint_cols].max(axis=1)
    return df, hp_cols


def plot_oat_sensitivity(
    results,
    threshold=0.1,
    titles=None,
    save_path=None,
):
    """
    results: { method: { hp_name: { exp_id: df } } }
    One figure per method. Each figure has K subplots (one per HP).
    Each subplot has two panels: loss (top) and max_constraint (bottom).
    6 curves per subplot, one per experiment.
    """
    colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
        "#9C755F", "#BAB0AB",
    ]
    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    for method, hp_dict in results.items():
        hp_names = list(hp_dict.keys())
        n_hps = len(hp_names)
        if n_hps == 0:
            continue

        fig, axes = plt.subplots(
            2, n_hps,
            figsize=(4 * n_hps, 6),
            sharex='col',
        )
        # ensure 2D even for single HP
        if n_hps == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle(f"OAT Sensitivity — {method}", fontsize=13, y=1.01)

        for col, hp in enumerate(hp_names):
            ax_loss = axes[0, col]
            ax_con  = axes[1, col]
            exp_dict = hp_dict[hp]

            for j, (exp_id, df) in enumerate(exp_dict.items()):
                color  = colors[j % len(colors)]
                marker = marker_styles[j % len(marker_styles)]
                label  = titles[j] if titles else str(exp_id)

                x = df[hp].values

                # ── loss panel ───────────────────────────────────────────────
                ax_loss.plot(x, df['loss'].values, lw=2.2, color=color,
                             label=label)
                ax_loss.plot(x, df['loss'].values, marker, color=color,
                             markersize=6, alpha=0.8)

                # ── constraint panel ─────────────────────────────────────────
                ax_con.plot(x, df['max_constraint'].values, lw=2.2,
                            color=color, label=label)
                ax_con.plot(x, df['max_constraint'].values, marker,
                            color=color, markersize=6, alpha=0.8)

            # ── threshold line + infeasible shading ──────────────────────────
            ax_con.axhline(threshold, color='red', lw=1.5,
                           linestyle='--', label=f'threshold={threshold}')
            ax_con.axhspan(threshold, ax_con.get_ylim()[1],
                           color='red', alpha=0.07)

            # ── labels ───────────────────────────────────────────────────────
            ax_loss.set_title(hp, fontsize=10)
            ax_loss.set_ylabel("Loss" if col == 0 else "")
            ax_con.set_ylabel("Max Constraint" if col == 0 else "")
            ax_con.set_xlabel(hp, fontsize=9)

            ax_loss.grid(True, linestyle='--', alpha=0.35)
            ax_con.grid(True, linestyle='--', alpha=0.35)

            if col == n_hps - 1:
                ax_loss.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(Path(save_path) / f"oat_{method}.pdf",
                        bbox_inches='tight')
        plt.show()

def get_oat_sensitivity(data_dir):
    """
    For each method/experiment:
      - find best feasible config
      - fix all HPs at optimum
      - vary one HP at a time across its observed range
      - record loss + max_constraint at each value
    
    Returns: { method: { hp_name: { exp_id: df } } }
    df columns: [hp_name, loss, max_constraint, feasible]
    """
    results = {m: {} for m in METHODS}

    for method in METHODS:
        for exp in EXPERIMENTS:
            path = Path(data_dir) / f"{method}_{exp}.csv"
            df, hp_cols = load_last_epoch(path)  # df is ALL runs, not just feasible

            # ── find optimum from feasible runs ───────────────────────────────
            df_feasible = df[df['max_constraint'] <= THRESHOLD]
            if len(df_feasible) == 0:
                print(f"WARNING: no feasible runs — exp {exp}, {method}")
                continue
            best = df_feasible.sort_values('loss').iloc[0]
            optimal_hps = {hp: best[hp] for hp in hp_cols}

            # ── OAT: vary one HP, fix rest at optimal ─────────────────────────
            for hp in hp_cols:
                if df[hp].nunique() < 2:
                    continue

                records = []
                for hp_val in sorted(df[hp].dropna().unique()):
                    # fix all others at optimal, vary this one
                    mask = pd.Series(True, index=df.index)
                    for other_hp in hp_cols:
                        if other_hp == hp:
                            continue
                        mask &= (df[other_hp] == optimal_hps[other_hp])
                    mask &= (df[hp] == hp_val)

                    subset = df[mask]
                    if len(subset) == 0:
                        continue  # this combination wasn't run
                    
                    # take mean in case of multiple seeds
                    records.append({
                        hp: hp_val,
                        'loss': subset['loss'].mean(),
                        'max_constraint': subset['max_constraint'].mean()
                    })

                if not records:
                    continue

                if hp not in results[method]:
                    results[method][hp] = {}
                results[method][hp][exp] = pd.DataFrame(records)

    return results



if __name__ == "__main__":
    results = get_oat_sensitivity("./data/sensitivity_analysis/")

    # plot the results
    plot_oat_sensitivity(results, threshold=THRESHOLD,
                         save_path="./data/results/plots/")

    print(results)
