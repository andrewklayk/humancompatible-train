"""
Pareto front comparison: dual constrained optimizers vs. L2 regularization on the ACS fairness task.

Each constrained method is swept over eps (the demographic parity tolerance);
regularization is swept over lambda (the penalty coefficient).

Results are evaluated on a held-out test set and plotted as a Pareto front
(fairness violation vs. BCE loss).
"""

import csv
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSIncome, generate_categories
from fairret.statistic import PositiveRate
from torch.nn import Sequential
from torch.optim import AdamW

from humancompatible.train.dual_optim import ALM, iALM, PBM, nuPI

# ── config ──────────────────────────────────────────────────────────────────
SEEDS       = [0, 1, 2]
EPOCHS      = 50
BATCH_SIZE  = 256
LR_MODEL    = 1e-3
LR_DUAL     = 0.01

N_CONSTRAINTS = 5

REG_LAMBDAS  = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
# eps=0 means exact equality; PBM only supports inequalities so starts at 0.01
EPSILONS     = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
EPSILONS_PBM = [      0.01, 0.02, 0.05, 0.1, 0.2]

# (name, dual-optimizer factory: eps -> optimizer, epsilons, color, marker)
CONSTRAINED_METHODS = [
    ("ALM",  lambda eps: ALM(m=N_CONSTRAINTS,  lr=LR_DUAL,      is_ineq=(eps > 0)), EPSILONS,     "tab:orange", "s"),
    # ("iALM", lambda eps: iALM(m=N_CONSTRAINTS, beta=LR_DUAL,    is_ineq=(eps > 0)), EPSILONS,     "tab:green",  "^"),
    # ("nuPI", lambda eps: nuPI(m=N_CONSTRAINTS, ki=LR_DUAL,      is_ineq=(eps > 0)), EPSILONS,     "tab:red",    "D"),
    ("SPBM",  lambda eps: PBM(m=N_CONSTRAINTS, penalty_update='dimin_adapt', penalty_mult=0.999, gamma=0.25, penalty_range=(1., 2.,)), EPSILONS_PBM, "tab:purple", "v"),
]

torch.set_default_dtype(torch.float32)

PLOT_CONV_CB = 0.05
PLOT_CONV_L = 2
PLOT_CONV_DICT = {
    "ALM": [PLOT_CONV_CB],
    "SPBM": [PLOT_CONV_CB],
    "Regularization": [PLOT_CONV_L]
}

# ── data ────────────────────────────────────────────────────────────────────
def load_data():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data    = data_source.get_data(states=["FL"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories  = generate_categories(features=ACSIncome.features, definition_df=definition_df)
    df_feat, df_labels, _ = ACSIncome.df_to_pandas(acs_data, categories=categories, dummies=True)

    # sens_cols = ["SEX_Female", "SEX_Male"]
    sens_cols = [col for col in df_feat.columns if col.startswith("MAR_")]
    features  = df_feat.drop(columns=sens_cols).to_numpy(dtype=np.float32)
    labels    = df_labels.to_numpy(dtype=np.float32)
    groups    = df_feat[sens_cols].to_numpy(dtype=np.float32)

    # 70 / 15 / 15 split
    (X_tr, X_tmp, y_tr, y_tmp, g_tr, g_tmp) = train_test_split(
        features, labels, groups, test_size=0.3, random_state=42
    )
    (X_val, X_te, y_val, y_te, g_val, g_te) = train_test_split(
        X_tmp, y_tmp, g_tmp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr  = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    to_t = lambda a: torch.tensor(a)
    return (
        (to_t(X_tr),  to_t(y_tr),  to_t(g_tr)),
        (to_t(X_val), to_t(y_val), to_t(g_val)),
        (to_t(X_te),  to_t(y_te),  to_t(g_te)),
    )


# ── model ────────────────────────────────────────────────────────────────────
def make_model(n_features, seed):
    torch.manual_seed(seed)
    return Sequential(
        torch.nn.Linear(n_features, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32),         torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


# ── fairness metric ──────────────────────────────────────────────────────────
def pr_violations(logit, groups):
    preds = torch.sigmoid(logit)
    stats = PositiveRate()(preds, groups)
    pr_all = PositiveRate()(preds, sens=None)
    return torch.abs(stats - pr_all)


# ── evaluation ───────────────────────────────────────────────────────────────
criterion = torch.nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, X, y, groups):
    logit = model(X)
    loss  = criterion(logit, y).item()
    violations = pr_violations(logit, groups).tolist()  # list of N_CONSTRAINTS floats
    return loss, violations


# ── training loops ───────────────────────────────────────────────────────────
def _make_epoch_row(epoch, tr_batch_losses, tr_batch_viols, te_loss, te_viols, wall_time):
    tr_arr = np.array(tr_batch_viols)  # (n_batches, N_CONSTRAINTS)
    row = {"epoch": epoch, "wall_time": wall_time,
           "tr_loss": np.mean(tr_batch_losses), "tr_viol_max": tr_arr.mean(axis=0).max()}
    for i in range(N_CONSTRAINTS):
        row[f"tr_viol_{i}"] = tr_arr[:, i].mean()
    row["te_loss"] = te_loss
    row["te_viol_max"] = max(te_viols)
    for i in range(N_CONSTRAINTS):
        row[f"te_viol_{i}"] = te_viols[i]
    return row


def train_regularized(X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te, lam, seed):
    model     = make_model(X_tr.shape[1], seed)
    optimizer = AdamW(model.parameters(), lr=LR_MODEL)
    dataset   = torch.utils.data.TensorDataset(X_tr, g_tr, y_tr)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            generator=torch.Generator().manual_seed(seed))

    epoch_log = []
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        model.train()
        tr_batch_losses, tr_batch_viols = [], []
        for xb, gb, yb in loader:
            optimizer.zero_grad()
            logit = model(xb)
            loss  = criterion(logit, yb) + lam * pr_violations(logit, gb).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                tr_batch_losses.append(criterion(logit, yb).item())  # pure BCE, not penalized
                tr_batch_viols.append(pr_violations(logit, gb).tolist())
        wall_time = time.perf_counter() - t0

        model.eval()
        te_loss_ep, te_viols_ep = evaluate(model, X_te, y_te, g_te)
        epoch_log.append(_make_epoch_row(epoch, tr_batch_losses, tr_batch_viols, te_loss_ep, te_viols_ep, wall_time))

    te_loss,  te_viols  = te_loss_ep, te_viols_ep  # reuse last epoch's test eval
    val_loss, val_viols = evaluate(model, X_val, y_val, g_val)
    tr_loss,  tr_viols  = evaluate(model, X_tr,  y_tr,  g_tr)
    return te_loss, te_viols, val_loss, val_viols, tr_loss, tr_viols, epoch_log


def train_constrained(make_dual, X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te, eps, seed):
    model      = make_model(X_tr.shape[1], seed)
    optimizer  = AdamW(model.parameters(), lr=LR_MODEL)
    dual_optim = make_dual(eps)
    dataset    = torch.utils.data.TensorDataset(X_tr, g_tr, y_tr)
    loader     = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                             generator=torch.Generator().manual_seed(seed))

    epoch_log = []
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        model.train()
        tr_batch_losses, tr_batch_viols = [], []
        for xb, gb, yb in loader:
            optimizer.zero_grad()
            logit      = model(xb)
            loss       = criterion(logit, yb)
            constraint = pr_violations(logit, gb) - eps
            lagr       = dual_optim.forward_update(loss, constraint)
            lagr.backward()
            optimizer.step()
            with torch.no_grad():
                tr_batch_losses.append(loss.item())
                tr_batch_viols.append((constraint + eps).tolist())  # = pr_violations
        wall_time = time.perf_counter() - t0

        model.eval()
        te_loss_ep, te_viols_ep = evaluate(model, X_te, y_te, g_te)
        epoch_log.append(_make_epoch_row(epoch, tr_batch_losses, tr_batch_viols, te_loss_ep, te_viols_ep, wall_time))

    te_loss,  te_viols  = te_loss_ep, te_viols_ep
    val_loss, val_viols = evaluate(model, X_val, y_val, g_val)
    tr_loss,  tr_viols  = evaluate(model, X_tr,  y_tr,  g_tr)
    return te_loss, te_viols, val_loss, val_viols, tr_loss, tr_viols, epoch_log


# ── main ─────────────────────────────────────────────────────────────────────
def _summarise_viols(arr):
    """arr: (n_seeds, N_CONSTRAINTS) → dict of mean/std keys."""
    d = {"fair_max_mean": np.mean(arr.max(axis=1)), "fair_max_std": np.std(arr.max(axis=1))}
    for i in range(N_CONSTRAINTS):
        d[f"fair_{i}_mean"] = np.mean(arr[:, i])
        d[f"fair_{i}_std"]  = np.std(arr[:, i])
    return d

def sweep(train_fn, hparams, X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te,
          log_dir=None, run_name=None):
    results = []
    for hp in hparams:
        te_losses, te_vs, val_losses, val_vs, tr_losses, tr_vs, ep_times = [], [], [], [], [], [], []
        for seed in SEEDS:
            print(f" Start train on seed {seed}")
            te_loss, te_viols, val_loss, val_viols, tr_loss, tr_viols, epoch_log = train_fn(
                X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te, hp, seed
            )
            if log_dir and run_name:
                save_epoch_log(epoch_log, log_dir, run_name, hp, seed)
            te_losses.append(te_loss);   te_vs.append(te_viols)
            val_losses.append(val_loss); val_vs.append(val_viols)
            tr_losses.append(tr_loss);   tr_vs.append(tr_viols)
            ep_times.append(np.mean([row["wall_time"] for row in epoch_log]))

        te_arr  = np.array(te_vs)   # (n_seeds, N_CONSTRAINTS)
        val_arr = np.array(val_vs)
        tr_arr  = np.array(tr_vs)

        r = {"hp": hp,
             "loss_mean":       np.mean(te_losses),  "loss_std":       np.std(te_losses),
             "val_loss_mean":   np.mean(val_losses),  "val_loss_std":   np.std(val_losses),
             "tr_loss_mean":    np.mean(tr_losses),   "tr_loss_std":    np.std(tr_losses),
             "epoch_time_mean": np.mean(ep_times),    "epoch_time_std": np.std(ep_times),
             **_summarise_viols(te_arr),
             **{f"val_{k}": v for k, v in _summarise_viols(val_arr).items()},
             **{f"tr_{k}":  v for k, v in _summarise_viols(tr_arr).items()},
        }
        results.append(r)
        print(f"  hp={hp:.4g}  "
              f"test loss={r['loss_mean']:.4f} max_viol={r['fair_max_mean']:.4f}  |  "
              f"val  loss={r['val_loss_mean']:.4f} max_viol={r['val_fair_max_mean']:.4f}  |  "
              f"train loss={r['tr_loss_mean']:.4f} max_viol={r['tr_fair_max_mean']:.4f}  |  "
              f"epoch time={r['epoch_time_mean']:.2f}s±{r['epoch_time_std']:.2f}s")
    return results


def _draw_pareto(ax, all_series, loss_key, fair_key, loss_std_key, fair_std_key, title):
    for results, label, color, marker in all_series:
        fair   = [r[fair_key]     for r in results]
        loss   = [r[loss_key]     for r in results]
        fair_e = [r[fair_std_key] for r in results]
        loss_e = [r[loss_std_key] for r in results]
        hp     = [r["hp"]         for r in results]

        ax.errorbar(fair, loss, xerr=fair_e, yerr=loss_e,
                    fmt=marker+"-", color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.2, alpha=0.85)

        for x, y, h in zip(fair, loss, hp):
            ax.annotate(f"{h:.3g}", (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=color)

    ax.set_xlabel("max group |PR − mean PR|  (lower is fairer)")
    ax.set_ylabel("BCE loss  (lower is better)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _viol_fields(prefix):
    p = f"{prefix}_" if prefix else ""
    return (
        [f"{p}fair_max_mean", f"{p}fair_max_std"]
        + [f"{p}fair_{i}_mean" for i in range(N_CONSTRAINTS)]
        + [f"{p}fair_{i}_std"  for i in range(N_CONSTRAINTS)]
    )

RESULT_FIELDS = (
    ["hp", "loss_mean", "loss_std"] + _viol_fields("")
    + ["val_loss_mean", "val_loss_std"] + _viol_fields("val")
    + ["tr_loss_mean",  "tr_loss_std"]  + _viol_fields("tr")
    + ["epoch_time_mean", "epoch_time_std"]
)

LOG_EPOCH_FIELDS = (
    ["epoch", "wall_time", "tr_loss", "tr_viol_max"]
    + [f"tr_viol_{i}" for i in range(N_CONSTRAINTS)]
    + ["te_loss", "te_viol_max"]
    + [f"te_viol_{i}" for i in range(N_CONSTRAINTS)]
)

def save_epoch_log(epoch_log, log_dir, run_name, hp, seed):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{run_name}_hp{hp:.4g}_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_EPOCH_FIELDS)
        writer.writeheader()
        writer.writerows(epoch_log)

def _csv_path(name, results_dir):
    return f"{results_dir}/results_{name.replace(' ', '_')}.csv"

def save_method_results(results, name, results_dir="benchmark_results"):
    os.makedirs(results_dir, exist_ok=True)
    path = _csv_path(name, results_dir)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in RESULT_FIELDS})
    print(f"  Saved to {path}")

def load_method_results(name, results_dir="benchmark_results"):
    path = _csv_path(name, results_dir)
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        results = [{k: float(r[k]) for k in RESULT_FIELDS} for r in rows]
        print(f"  Loaded cached results from {path}")
        return results
    except FileNotFoundError:
        return None


def plot(all_series, out="pareto_fairness.svg"):
    fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    _draw_pareto(ax_tr, all_series,
                 "tr_loss_mean", "tr_fair_max_mean", "tr_loss_std", "tr_fair_max_std", "Train set")
    _draw_pareto(ax_te, all_series,
                 "loss_mean", "fair_max_mean", "loss_std", "fair_max_std", "Test set")

    fig.suptitle("Pareto front: constrained learning vs. regularization\n(ACS Income, demographic parity)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nSaved plot to {out}")


_COLOR_TO_CMAP = {
    "tab:blue":   plt.cm.Blues,
    "tab:orange": plt.cm.Oranges,
    "tab:green":  plt.cm.Greens,
    "tab:red":    plt.cm.Reds,
    "tab:purple": plt.cm.Purples,
}

def _load_epoch_logs_for_hp(method_name, hp, log_dir):
    """Returns {field: array(n_seeds, n_epochs)} or None if any file is missing."""
    seed_rows = []
    for seed in SEEDS:
        path = os.path.join(log_dir, f"{method_name}_hp{hp:.4g}_seed{seed}.csv")
        try:
            with open(path, newline="") as f:
                seed_rows.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)])
        except FileNotFoundError:
            return None
    fields = [k for k in seed_rows[0][0] if k != "epoch"]
    return {f: np.array([[row[f] for row in rows] for rows in seed_rows]) for f in fields}
    # each value: shape (n_seeds, n_epochs)


def plot_convergence(all_series, log_dir="benchmark_results/epoch_logs", out="convergence.svg",
                     hparams_filter=None, add_hline=PLOT_CONV_CB):
    """
    hparams_filter:
      None                        – plot all hp values (default)
      list[float]                 – keep only these hp values for every method
      dict[method_name, list]     – per-method filter; methods absent from the dict are unfiltered
    """
    from matplotlib.lines import Line2D

    metrics = (
        [("tr_loss",     "te_loss",     "Loss (BCE)"),
         ("tr_viol_max", "te_viol_max", "Max violation")]
        + [(f"tr_viol_{i}", f"te_viol_{i}", f"Violation {i}") for i in range(N_CONSTRAINTS)]
    )
    n_rows    = len(metrics)
    n_methods = len(all_series)

    fig, axes = plt.subplots(
        n_rows, n_methods,
        figsize=(4 * n_methods, 2.5 * n_rows),
        squeeze=False,
    )

    # loss row: share y within the row
    for col in range(1, n_methods):
        axes[0, col].sharey(axes[0, 0])

    # all violation rows (max + per-constraint): share a single y-axis
    viol_ref = axes[1, 0]
    for row in range(1, n_rows):
        for col in range(n_methods):
            if not (row == 1 and col == 0):
                axes[row, col].sharey(viol_ref)

    for col, (results, label, base_color, _marker) in enumerate(all_series):
        method_name = label.split(" (")[0]
        hparams     = [r["hp"] for r in results]

        if hparams_filter is not None:
            allowed = hparams_filter.get(method_name) if isinstance(hparams_filter, dict) else hparams_filter
            if allowed is not None:
                hparams = [hp for hp in hparams if hp in allowed]

        cmap        = _COLOR_TO_CMAP.get(base_color, plt.cm.Greys)
        hp_colors   = cmap(np.linspace(0.4, 0.9, max(len(hparams), 1)))

        for row, (tr_key, te_key, ylabel) in enumerate(metrics):
            ax = axes[row, col]

            for hp, color in zip(hparams, hp_colors):
                data = _load_epoch_logs_for_hp(method_name, hp, log_dir)
                if data is None:
                    continue
                epochs = np.arange(data[tr_key].shape[1])

                for key, color in [(tr_key, "tab:blue"), (te_key, "tab:orange")]:
                    mean = data[key].mean(axis=0)
                    std  = data[key].std(axis=0)
                    # lbl  = f"hp={hp:.3g}" if color == "dodgerblue" else None
                    lbl = f"Test" if color == "tab:orange" else "Train"
                    ax.plot(epochs, mean, color=color, linewidth=1.3, label=lbl)
                    ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

            ax.grid(True, alpha=0.3, linewidth=0.5)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)
            if row == 0:
                ax.set_title(method_name, fontsize=12, fontweight="bold")
                ax.legend(fontsize=10, loc="upper right", title_fontsize=6)
            if row == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=12)
            if row > 0:
                ax.hlines(y=add_hline, xmin=0, xmax=EPOCHS, ls='--', color='red')

    # fig.legend(
    #     # handles=[Line2D([0], [0], ls="-",  color="gray", label="test"),
    #     #          Line2D([0], [0], ls="--", color="gray", label="train")],
    #     loc="lower center", ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0),
    # )
    fig.suptitle("Convergence", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to {out}")


def run_or_load(name, train_fn, hparams, data, results_dir="benchmark_results"):
    """data: flat 9-tuple (X_tr,y_tr,g_tr, X_val,y_val,g_val, X_te,y_te,g_te)"""
    results = load_method_results(name, results_dir)
    if results is None:
        log_dir = os.path.join(results_dir, "epoch_logs")
        results = sweep(train_fn, hparams, *data, log_dir=log_dir, run_name=name)
        save_method_results(results, name, results_dir)
    return results


if __name__ == "__main__":
    data_loaded = False

    def get_data():
        global _data, data_loaded
        if not data_loaded:
            print("Loading data...")
            (X_tr, y_tr, g_tr), (X_val, y_val, g_val), (X_te, y_te, g_te) = load_data()
            print(f"Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}\n")
            _data = X_tr, y_tr, g_tr, X_val, y_val, g_val, X_te, y_te, g_te
            data_loaded = True
        return _data

    print("Regularization (λ)...")
    reg_results = run_or_load("Regularization", train_regularized, REG_LAMBDAS, get_data())

    all_series = [(reg_results, "Regularization", "tab:blue", "o")]

    for name, make_dual, epsilons, color, marker in CONSTRAINED_METHODS:
        print(f"\n{name} (ε)...")
        train_fn = lambda *args, md=make_dual: train_constrained(md, *args)
        results  = run_or_load(name, train_fn, epsilons, get_data())
        all_series.append((results, f"{name} (ε)", color, marker))

    plot(all_series)
    log_dir = os.path.join("benchmark_results", "epoch_logs")
    plot_convergence(all_series, log_dir=log_dir, hparams_filter=PLOT_CONV_DICT)
