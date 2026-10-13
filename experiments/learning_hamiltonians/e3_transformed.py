"""Experiments 2 and 3 -- observations are a distortion of (q,p), so the coordinates have to
be learned too (Sec. III B, Eqs. 12-17).

    python e3_transformed.py linear      # [x,y] = R(5 deg) diag(1,64) [q,p]
    python e3_transformed.py nonlinear   # Eq. (17): x = a + (b+a^2)^2, y = b + a^2

Encoder + decoder + Hhat are trained jointly. Derivatives reach the learned coordinates by
the chain rule (push_forward); f5 keeps the pair mutually inverse and f6 keeps the latent
space from collapsing, which is otherwise the cheapest way to satisfy f1 and f2.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from common import (H_true, pendulum_data, mlp, grad, push_forward, jac_det_2d,
                    train, plot_H, report, plt)

C = {"f1": 1.0, "f2": 10.0, "f3": 1.0, "f5": 1.0, "f6": 1e-6}


def observe(mode, q, p):
    """Map canonical (q,p) to what we actually get to see."""
    if mode == "linear":
        r = np.deg2rad(5.0)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        A = R @ np.diag([1.0, 64.0])
        return np.stack([q, p], 1) @ A.T, A
    a, b = q / 20.0, p / 10.0
    return np.stack([a + (b + a**2) ** 2, b + a**2], 1), None


def main(mode="linear"):
    torch.manual_seed(0)
    q, p, qd, pd = pendulum_data()
    XY, A = observe(mode, q, p)
    # observed time derivatives, again by the chain rule on the true trajectory
    eps = 1e-5
    XYd = (observe(mode, q + eps * qd, p + eps * pd)[0] - XY) / eps

    X = torch.tensor(XY, dtype=torch.float32)
    Xd = torch.tensor(XYd, dtype=torch.float32)

    if mode == "linear":
        enc = torch.nn.Linear(2, 2, bias=False)
        dec = torch.nn.Linear(2, 2, bias=False)
    else:
        enc, dec = mlp([2, 32, 32, 2]), mlp([2, 32, 32, 2])
    H = mlp([2, 32, 32, 1])
    params = list(enc.parameters()) + list(dec.parameters()) + list(H.parameters())
    H0 = torch.tensor(H_true(q[0], p[0]), dtype=torch.float32)

    def loss_fn():
        z, zd = push_forward(enc, X, Xd)
        dH = grad(H(z), z)
        f1 = ((dH[:, 1] - zd[:, 0]) ** 2).mean()       # dH/dp_hat = q_hat_dot
        f2 = ((dH[:, 0] + zd[:, 1]) ** 2).mean()       # dH/dq_hat = -p_hat_dot
        f3 = (H(z[:1]).squeeze() - H0) ** 2
        f5 = ((dec(z) - X) ** 2).mean()
        f6 = (jac_det_2d(dec, z) ** -2).mean()
        return (C["f1"] * f1 + C["f2"] * f2 + C["f3"] * f3
                + C["f5"] * f5 + C["f6"] * f6)

    train(loss_fn, params, steps=6000)

    with torch.no_grad():
        z = enc(X)
        H_hat = H(z).squeeze().numpy()
    H_ref = H_true(q, p)
    print(f"\n{mode} distortion:")
    H_hat = report(H_hat, H_ref, "Hhat vs true H")

    figs = Path(__file__).parent / "figs"
    plot_H(q, p, H_hat, H_ref, f"{mode} distortion", figs / f"e3_{mode}_H.png")

    # identifiability check: does the recovered map keep q unmixed? (Sec. III B, Eqs. 13-16)
    if mode == "linear":
        S = enc.weight.detach().numpy() @ A
        print("  composite map S = A_hat . A  (true (q,p) -> learned coords):")
        print("   ", np.array2string(S, precision=4).replace("\n", "\n    "))
        print(f"  off-diagonal / diagonal magnitude: "
              f"{abs(S[0, 1]) / abs(S[0, 0]):.3f}, {abs(S[1, 0]) / abs(S[1, 1]):.3f}")
    zn = z.numpy()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for j, lbl in enumerate([r"$\hat{q}$", r"$\hat{p}$"]):
        ax[j].plot(q, zn[:, j], ".", ms=2, label="vs q")
        ax[j].plot(p, zn[:, j], ".", ms=2, label="vs p")
        ax[j].set(xlabel="true coordinate", ylabel=lbl)
        ax[j].legend(markerscale=5)
    fig.suptitle(f"{mode}: one true coordinate should map monotonically into one latent one")
    fig.tight_layout()
    fig.savefig(figs / f"e3_{mode}_coords.png", dpi=130)
    print(f"  wrote {figs / f'e3_{mode}_coords.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "linear")
