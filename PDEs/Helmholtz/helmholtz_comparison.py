"""
Helmholtz PINN benchmark: AdamW+penalty vs ALM vs PBM.

PDE: Δu + k²u = q on [-1,1]², u = 0 on boundary.
Analytic solution: u = sin(a1·π·x)·sin(a2·π·y), k=1, a1=1, a2=4.

Runs each method with N seeds and reports mean ± std curves.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from humancompatible.train.dual_optim import ALM, PBM

# ── PDE constants ─────────────────────────────────────────────────────────────
k, a1, a2 = 1, 1, 4
N_TEST = 61  # holdout test grid resolution (different from training grid)


def source_term(pts: torch.Tensor) -> torch.Tensor:
    x, y = pts[:, 0:1], pts[:, 1:2]
    s = torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)
    return (-(a1 * np.pi) ** 2 - (a2 * np.pi) ** 2 + k ** 2) * s


def analytic_solution(pts: torch.Tensor) -> torch.Tensor:
    x, y = pts[:, 0:1], pts[:, 1:2]
    return torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)


def pde_residual(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(u)
    g = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0:1], x, grad_outputs=ones, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(g[:, 1:2], x, grad_outputs=ones, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy + k ** 2 * u - source_term(x)


# ── Network ───────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, width: int = 128, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def make_data(device: torch.device, n_grid: int = 51):
    coords = np.mgrid[-1:1:complex(0, n_grid), -1:1:complex(0, n_grid)].reshape(2, -1).T
    X_coll = torch.tensor(coords, dtype=torch.float32, device=device)

    on_bdry = (
        (X_coll[:, 0] == -1) | (X_coll[:, 0] == 1) |
        (X_coll[:, 1] == -1) | (X_coll[:, 1] == 1)
    )
    X_bdry = X_coll[on_bdry]
    u_bdry = torch.zeros(X_bdry.shape[0], 1, device=device)

    # holdout test points: fine grid at N_TEST resolution (not aligned with training grid)
    coords_test = np.mgrid[-1:1:complex(0, N_TEST), -1:1:complex(0, N_TEST)].reshape(2, -1).T
    X_test = torch.tensor(coords_test, dtype=torch.float32, device=device)
    y_test = analytic_solution(X_test)

    return X_coll, X_bdry, u_bdry, X_test, y_test


# ── Training step ─────────────────────────────────────────────────────────────
def train_step(
    model: PINN,
    optimizer: torch.optim.Optimizer,
    X_coll: torch.Tensor,
    X_bdry: torch.Tensor,
    u_bdry: torch.Tensor,
    loss_f: nn.Module,
    beta: float,
    dual_opt,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    X_v = X_coll.detach().requires_grad_(True)
    u = model(X_v)
    pde_loss = loss_f(pde_residual(u, X_v), torch.zeros_like(u))

    u_bdry_pred = model(X_bdry)
    bc_loss = loss_f(u_bdry_pred, u_bdry) - 1e-4

    if dual_opt is None:
        (pde_loss + beta * bc_loss).backward()
    elif isinstance(dual_opt, PBM):
        dual_opt.forward_update(pde_loss, bc_loss.unsqueeze(0)).backward()
    else:
        dual_opt.forward_update(pde_loss, bc_loss).backward()

    optimizer.step()
    return pde_loss.item(), bc_loss.item() + 1e-4


@torch.no_grad()
def relative_l2(model: PINN, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    model.eval()
    pred = model(X_test)
    return (torch.linalg.norm(pred - y_test) / torch.linalg.norm(y_test)).item()


# ── Single experiment run ─────────────────────────────────────────────────────
def run(method: str, seed: int, args, device: torch.device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_coll, X_bdry, u_bdry, X_test, y_test = make_data(device, args.n_grid)
    model = PINN(args.width, args.depth).to(device)
    loss_f = nn.MSELoss()
    dual_opt = None

    if method == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    elif method == "alm":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        dual_opt = ALM(
            m=1,
            lr=args.dual_lr,
            is_ineq=True,
            device=device,
        )

    elif method == "pbm":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        dual_opt = PBM(
            m=1,
            penalty_update="const",
            pbf="quadratic_logarithmic",
            gamma=0.1,
            penalty_mult=0.999,
            init_duals=0.01,
            init_penalties=1.0,
            penalty_range=(0.5, 2.0),
            dual_range=(0.01, 100.0),
            delta=1.0,
            device=device,
        )

    history: dict[str, list] = {"pde_loss": [], "bc_violation": [], "test_err": []}

    for _ in range(args.epochs):
        pde_loss, bc_loss = train_step(
            model, optimizer, X_coll, X_bdry, u_bdry, loss_f, args.beta, dual_opt
        )
        history["pde_loss"].append(pde_loss)
        history["bc_violation"].append(bc_loss)
        history["test_err"].append(relative_l2(model, X_test, y_test))

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: dict, args, out_path: str) -> None:
    methods = list(results.keys())
    colors = {"adamw": "tab:blue", "alm": "tab:orange", "pbm": "tab:green"}
    labels = {
        "adamw": f"AdamW+penalty (β={args.beta})",
        "alm":   "ALM",
        "pbm":   "PBM",
    }
    metrics = [
        ("pde_loss",     "PDE Residual Loss (MSE)"),
        ("bc_violation", "BC Violation (MSE)"),
        ("test_err",     "Test Relative L2 Error"),
    ]
    epochs_range = np.arange(1, args.epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, ylabel) in zip(axes, metrics):
        for method in methods:
            arr = results[method][key]          # shape (n_seeds, epochs)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            # ax.plot(epochs_range, mean, label=labels[method], color=colors[method])
            ax.semilogy(epochs_range, mean, label=labels[method], color=colors[method])
            ax.fill_between(
                epochs_range,
                np.maximum(mean - std, 1e-10),
                mean + std,
                alpha=0.2,
                color=colors[method],
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs",   type=int,   default=4000)
    parser.add_argument("--lr",       type=float, default=1e-3,  help="Primal learning rate")
    parser.add_argument("--dual_lr",  type=float, default=0.001,  help="Dual learning rate (ALM)")
    parser.add_argument("--beta",     type=float, default=10.0,  help="Penalty weight for AdamW")
    parser.add_argument("--width",    type=int,   default=128,   help="Hidden layer width")
    parser.add_argument("--depth",    type=int,   default=4,     help="Number of hidden layers")
    parser.add_argument("--n_grid",   type=int,   default=51,    help="Collocation grid resolution")
    parser.add_argument("--seeds",    type=int,   nargs="+",     default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods",  type=str,   nargs="+",     default=["adamw", "alm", "pbm"],
                        choices=["adamw", "alm", "pbm"])
    parser.add_argument("--out",    type=str, default="./PDEs/Helmholtz/helmholtz_comparison.png")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Epochs: {args.epochs} | Seeds: {args.seeds}")

    results: dict[str, np.ndarray] = {}
    for method in args.methods:
        runs = []
        for seed in args.seeds:
            print(f"  [{method.upper()}] seed={seed} ...", flush=True)
            runs.append(run(method, seed, args, device))
        results[method] = {
            key: np.array([r[key] for r in runs])  # (n_seeds, epochs)
            for key in runs[0]
        }

    plot_results(results, args, args.out)

    print("\n=== Best test-error summary (mean ± std over seeds) ===")
    labels = {
        "adamw": f"AdamW (β={args.beta})",
        "alm":   "ALM",
        "pbm":   "PBM",
    }
    header = f"{'Method':<22} {'PDE Loss':>22} {'BC Violation':>22} {'Test Rel-L2':>22}"
    print(header)
    print("-" * len(header))
    for method in args.methods:
        arr_test = results[method]["test_err"]          # (n_seeds, epochs)
        best_ep  = arr_test.argmin(axis=1)              # epoch index per seed
        seed_idx = np.arange(len(args.seeds))
        row = []
        for key in ["pde_loss", "bc_violation", "test_err"]:
            vals = results[method][key][seed_idx, best_ep]  # value at best epoch
            row.append(f"{vals.mean():.4e} ± {vals.std():.4e}")
        print(f"{labels[method]:<22} {row[0]:>22} {row[1]:>22} {row[2]:>22}")


if __name__ == "__main__":
    main()
