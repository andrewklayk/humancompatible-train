"""Experiment 4 -- learn H from rendered pendulum video (Sec. III C).

A single frame of a pendulum is not a state: it shows q but not p. The paper's fix is to
drag a decaying tail of past positions behind the bob, so one frame carries both. Pipeline:
frames -> 20 principal components -> dense autoencoder -> 4-d latent (q1,q2,p1,p2) -> Hhat.

Collapse is the danger here (a constant Hhat on a collapsed latent satisfies every residual),
so the paper adds a std-dev penalty on Hhat and max-squared-error terms on top of f1, f2.
"""

import numpy as np
import torch
from pathlib import Path
from common import H_true, trajectories, finite_diff, mlp, grad, push_forward, train, plt

SIZE, TAIL, DECAY, SIGMA = 24, 4, 0.5, 0.18
STRIDE, N_PC, LATENT = 10, 20, 4
C = {"f1": 1.0, "f2": 10.0, "f3": 1.0, "f5": 1.0, "std": 1.0, "max": 0.1}


def render(qs):
    """qs: (n_frames, TAIL) of current + previous angles -> flat images (n_frames, SIZE**2)."""
    g = np.linspace(-1.35, 1.35, SIZE)
    xx, yy = np.meshgrid(g, g)
    img = np.zeros((len(qs), SIZE, SIZE))
    for k in range(TAIL):
        bx, by = np.sin(qs[:, k]), -np.cos(qs[:, k])
        d2 = (xx - bx[:, None, None]) ** 2 + (yy - by[:, None, None]) ** 2
        img += DECAY**k * np.exp(-d2 / SIGMA**2)
    return img.reshape(len(qs), -1)


def make_video():
    """Frames, their time derivatives (central differences), and the true H for reference."""
    Q, P = trajectories(n_traj=40, n_steps=400)
    Q, P = Q[:, ::STRIDE], P[:, ::STRIDE]
    dt = 0.01 * STRIDE
    # stack each frame's tail of past angles, then drop the frames that have no tail
    qs = np.stack([Q[:, TAIL - 1 - k:Q.shape[1] - k] for k in range(TAIL)], -1)
    n_traj, T, _ = qs.shape
    frames = render(qs.reshape(-1, TAIL)).reshape(n_traj, T, -1)
    # difference each pixel along time: flatten to (n_traj * n_pixels, T) first
    Fd, F = finite_diff(frames.transpose(0, 2, 1).reshape(-1, T), dt)
    flat = lambda A: A.reshape(n_traj, SIZE**2, T - 2).transpose(0, 2, 1).reshape(-1, SIZE**2)
    H_ref = H_true(Q[:, TAIL - 1:][:, 1:-1], P[:, TAIL - 1:][:, 1:-1])
    return flat(F), flat(Fd), H_ref.ravel()


def main():
    torch.manual_seed(0)
    F, Fd, H_ref = make_video()
    print(f"{len(F)} frames of {SIZE}x{SIZE}")

    # PCA: linear, so the same projection carries the frame derivatives
    mu = F.mean(0)
    _, _, Vt = np.linalg.svd(F - mu, full_matrices=False)
    W = Vt[:N_PC].T
    X = torch.tensor((F - mu) @ W, dtype=torch.float32)
    Xd = torch.tensor(Fd @ W, dtype=torch.float32)
    s = X.std(0)                          # whiten; the latent scale is arbitrary anyway
    X, Xd = X / s, Xd / s

    enc = mlp([N_PC, 64, 64, LATENT])
    dec = mlp([LATENT, 64, 64, N_PC])
    H = mlp([LATENT, 64, 64, 1])
    params = list(enc.parameters()) + list(dec.parameters()) + list(H.parameters())
    H0 = torch.tensor(H_ref[0], dtype=torch.float32)
    n = LATENT // 2

    def loss_fn():
        z, zd = push_forward(enc, X, Xd)
        dH = grad(H(z), z)
        r1 = (dH[:, n:] - zd[:, :n]) ** 2      # dH/dp_hat = q_hat_dot
        r2 = (dH[:, :n] + zd[:, n:]) ** 2      # dH/dq_hat = -p_hat_dot
        Hz = H(z).squeeze()
        return (C["f1"] * r1.mean() + C["f2"] * r2.mean()
                + C["f3"] * (Hz[0] - H0) ** 2
                + C["f5"] * ((dec(z) - X) ** 2).mean()
                + C["std"] * torch.relu(1.0 - Hz.std()) ** 2   # don't let Hhat go constant
                + C["max"] * (r1.max() + r2.max()))

    train(loss_fn, params, steps=6000, lr=3e-3)

    z, zd = push_forward(enc, X, Xd)
    Hz = H(z).squeeze()
    dHdt = (grad(Hz, z) * zd).sum(1).detach().numpy()
    H_hat = Hz.detach().numpy()
    print(f"\n  |dHhat/dt|  mean {np.abs(dHdt).mean():.3e}   max {np.abs(dHdt).max():.3e}")
    print(f"  spread of Hhat: std {H_hat.std():.3e}  (a collapsed solution has std ~ 0)")
    print(f"  corr(Hhat, true H): {np.corrcoef(H_hat, H_ref)[0, 1]:.4f}")

    figs = Path(__file__).parent / "figs"
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].imshow(F[0].reshape(SIZE, SIZE), cmap="magma")
    ax[0].set(title="frame (bob + decaying tail)", xticks=[], yticks=[])
    ax[1].plot(H_ref, H_hat, ".", ms=2)
    ax[1].set(xlabel="true H", ylabel=r"$\hat{H}$", title="learned vs true")
    ax[2].hist(dHdt, bins=60)
    ax[2].set(xlabel=r"$d\hat{H}/dt$", title="conservation along trajectories")
    fig.tight_layout()
    fig.savefig(figs / "e4_video.png", dpi=130)
    print(f"  wrote {figs / 'e4_video.png'}")


if __name__ == "__main__":
    main()
