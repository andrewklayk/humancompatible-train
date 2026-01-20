import torch
from humancompatible.train.optim.PBM import PBM

def plot_balls_trajectory(trajectory):
    """
    trajectory: array-like of shape (N, 2), where each row is [x, y]
    """

    trajectory = np.asarray(trajectory)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Feasible regions: unit balls
    ball_centers = [(-2, 0), (2, 0)]
    radius = 1.0
    labels = [r"$g_1(x,y) \leq 0 $", r"$g_2(x,y) \leq 0$"]

    for center, label in zip(ball_centers, labels):
        ball = Circle(
            center,
            radius=radius,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(ball)
        # Label inside the ball
        ax.text(
            center[0], center[1] - 0.5,
            label,
            fontsize=10,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            zorder=5
        )

    # Trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="black",
        linewidth=2.0,
        zorder=3,
    )

    # Emphasize x_0 and x_n
    x0 = trajectory[0]
    xn = trajectory[-1]

    ax.scatter(
        x0[0], x0[1],
        s=80,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=2,
        zorder=4,
    )
    ax.scatter(
        xn[0], xn[1],
        s=80,
        marker="s",
        facecolor="black",
        edgecolor="black",
        zorder=4,
    )

    # Labels for x_0 and x_n
    ax.annotate(
        r"$x_0$",
        xy=(x0[0], x0[1]),
        xytext=(-8, 8),
        textcoords="offset points",
        fontsize=12,
        zorder=5,
    )
    ax.annotate(
        r"$x_n$",
        xy=(xn[0], xn[1]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=12,
        zorder=5,
    )

    # Heatmap for x^2 + y^2
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X*2 + Y*2

    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.5, zorder=0)
    plt.colorbar(contour, ax=ax, label=r'$x^2 + y^2$')

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    plt.show()


def balls(x, sample):
    g1 = ((x[0] - 2 + sample)*2 + x[1]*2 - 1)
    g2 = ((x[0] + 2 + sample)*2 + x[1]*2 - 1)
    # g1 = ((x[0] - 2)*2 + x[1]*2 - 1) + sample
    # g2 = ((x[0] + 2)*2 + x[1]*2 - 1) + sample
    if g1 <= g2:
        return g1
    else:
        return g2

def parabola(x):
    return x[0]*2 + x[1]*2

xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
with torch.no_grad():
    xy[0] = 0
    xy[1] = 1

pbm = PBM([xy], m=1, lr=0.1, dual_bounds=(1e-3, 1e6), penalty_update_m='DIMINISH', epoch_len=2, mu=0)

samples = [
    # torch.tensor([0]),
    # torch.tensor([0])
    torch.tensor([-0.1]),
    torch.tensor([0.1])
]


iters = 100

param_log = []
con_log = []
dual_log = []
c_grad_log = []

for i in range(iters):
    param_log.append(
        xy.detach().numpy().copy()
    )
    # print(xy)
    
    minibatch = samples[i % 2]
    
    c = balls(xy, minibatch) #+ minibatch
    c_grad = torch.autograd.grad(c, xy)
    c_grad_log.append(c_grad[0].detach().numpy())
    
    pbm.dual_step(0, c)
    dual_log.append(pbm._dual_vars.detach().numpy().copy().item())

    obj = parabola(xy)

    pbm.step(obj)

    con_log.append(c.detach().numpy().copy().item())# - minibatch)


# print(param_log)
# print(con_log)

from matplotlib import pyplot as plt
import numpy as np


plot_balls_trajectory(
    np.array(param_log)
)

print(np.array(dual_log))
print(np.array(con_log))
print(np.array(c_grad_log))