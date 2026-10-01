"""Shared pieces for the Bertalan et al. (2019) reproductions.

Deliberately small: pendulum data, a tanh MLP, and three autograd helpers.
Nothing here imports from `humancompatible` -- these scripts stand alone.
"""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DT = 0.01


def H_true(q, p):
    """H(q,p) = p^2/2 + (1 - cos q)   -- Eq. (5), the nonlinear pendulum."""
    return p**2 / 2 + (1 - np.cos(q))


def trajectories(n_traj=25, n_steps=150, dt=DT, seed=0):
    """Symplectic Euler on qdot = p, pdot = -sin q. Returns (q, p), each (n_traj, n_steps)."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-2.0, 2.0, n_traj)
    p = rng.uniform(-1.5, 1.5, n_traj)
    Q, P = [q.copy()], [p.copy()]
    for _ in range(n_steps - 1):
        p = p - dt * np.sin(q)  # momentum first -> symplectic
        q = q + dt * p
        Q.append(q.copy())
        P.append(p.copy())
    return np.array(Q).T, np.array(P).T


def finite_diff(Z, dt=DT):
    """Central differences along the time axis. Returns (Zdot, Z) with endpoints dropped."""
    return (Z[:, 2:] - Z[:, :-2]) / (2 * dt), Z[:, 1:-1]


def pendulum_data(dt=DT, **kw):
    """Flat scattered samples (q, p, qdot, pdot) taken off trajectories, derivatives by
    finite differences -- i.e. everything comes from the data, not from the known ODE."""
    Q, P = trajectories(dt=dt, **kw)
    Qd, Q = finite_diff(Q, dt)
    Pd, P = finite_diff(P, dt)
    return [a.ravel() for a in (Q, P, Qd, Pd)]


# --- torch helpers ------------------------------------------------------------

def mlp(sizes, act=torch.nn.Tanh):
    """Plain MLP, tanh inside, identity on the output -- Eq. (9)."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), act()]
    return torch.nn.Sequential(*layers[:-1])


def grad(y, x):
    """d(sum y)/dx, keeping the graph so it can be differentiated again."""
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def push_forward(enc, x, xdot):
    """Chain rule through the encoder: z = enc(x), zdot = D(enc)(x) . xdot."""
    x = x.detach().requires_grad_(True)
    z = enc(x)
    zdot = torch.stack([(grad(z[:, j], x) * xdot).sum(1) for j in range(z.shape[1])], 1)
    return z, zdot


def jac_det_2d(f, z):
    """det of the 2x2 Jacobian of f at each row of z (the f6 anti-collapse term).

    z must already be in the autograd graph, so the penalty also reaches the encoder.
    """
    out = f(z)
    g0, g1 = grad(out[:, 0], z), grad(out[:, 1], z)
    return g0[:, 0] * g1[:, 1] - g0[:, 1] * g1[:, 0]


def train(loss_fn, params, steps=4000, lr=1e-2, decay=0.999, log_every=1000):
    """Full-batch Adam with an exponentially decaying LR."""
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, decay)
    for i in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (i % log_every == 0 or i == steps - 1):
            print(f"  step {i:5d}   loss {loss.item():.3e}")
    return loss.item()


# --- reporting ----------------------------------------------------------------

def report(H_hat, H_ref, name):
    """H is only identified up to an additive constant; align on the mean before comparing."""
    H_hat = H_hat - H_hat.mean() + H_ref.mean()
    err = np.abs(H_hat - H_ref)
    print(f"  {name}: mean |err| {err.mean():.3e}   max |err| {err.max():.3e}   "
          f"corr {np.corrcoef(H_hat, H_ref)[0, 1]:.5f}")
    return H_hat


def plot_H(q, p, H_hat, H_ref, title, path):
    """Left: learned H over phase space. Right: learned vs true, point by point."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    s = ax[0].scatter(q, p, c=H_hat, s=4, cmap="viridis")
    ax[0].set(xlabel="q", ylabel="p", title=title)
    fig.colorbar(s, ax=ax[0], label=r"$\hat{H}$")
    ax[1].plot(H_ref, H_hat, ".", ms=2)
    lim = [H_ref.min(), H_ref.max()]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set(xlabel="true H", ylabel=r"learned $\hat{H}$", title="aligned on the mean")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")
