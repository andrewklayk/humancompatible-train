"""Experiment 1b -- recover H for the pendulum with a network (Sec. III, Eqs. 10-11).

Observations are already canonical, so there is no autoencoder: just Hhat(q,p) trained on
f1, f2, f3. Run with `--ablate f1` (or f2 / f4) to reproduce the ablation of Sec. III: f4 is
formally redundant and dropping it costs nothing, but dropping f1 or f2 alone breaks
training even though theory says the other two terms should suffice.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from common import H_true, pendulum_data, mlp, grad, train, plot_H, report

C = {"f1": 1.0, "f2": 10.0, "f3": 1.0, "f4": 0.0}  # weights from the paper


def main(ablate=None):
    torch.manual_seed(0)
    q, p, qd, pd = pendulum_data()
    Z = torch.tensor(np.stack([q, p], 1), dtype=torch.float32)
    QD = torch.tensor(qd, dtype=torch.float32)
    PD = torch.tensor(pd, dtype=torch.float32)

    c = dict(C)
    if ablate:
        c[ablate] = 0.0
        if ablate in ("f1", "f2"):
            c["f4"] = 1.0  # the term theory says should stand in for it
    print(f"weights: {c}")

    H = mlp([2, 32, 32, 1])
    H0 = torch.tensor(H_true(q[0], p[0]), dtype=torch.float32)
    z0 = Z[:1]

    def loss_fn():
        z = Z.detach().requires_grad_(True)
        dH = grad(H(z), z)
        dHdq, dHdp = dH[:, 0], dH[:, 1]
        f1 = ((dHdp - QD) ** 2).mean()
        f2 = ((dHdq + PD) ** 2).mean()
        f3 = (H(z0).squeeze() - H0) ** 2
        f4 = ((dHdq * QD + dHdp * PD) ** 2).mean()  # (dH/dt)^2
        return c["f1"] * f1 + c["f2"] * f2 + c["f3"] * f3 + c["f4"] * f4

    train(loss_fn, H.parameters(), steps=4000)

    with torch.no_grad():
        H_hat = H(Z).squeeze().numpy()
    H_ref = H_true(q, p)
    tag = f"NN (ablate {ablate})" if ablate else "NN"
    H_hat = report(H_hat, H_ref, tag)
    name = f"e2_nn{'_no_' + ablate if ablate else ''}.png"
    plot_H(q, p, H_hat, H_ref, tag, Path(__file__).parent / "figs" / name)


if __name__ == "__main__":
    a = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--ablate" else None
    main(a)
