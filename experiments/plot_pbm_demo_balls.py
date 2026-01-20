from matplotlib.patches import Circle
import torch
from humancompatible.train.optim.PBM import PBM
import numpy as np
import matplotlib.pyplot as plt

def plot_balls_trajectory(trajectory_pbm, trajectory_sgd):
    """
    trajectory: array-like of shape (N, 2), where each row is [x, y]
    """

    trajectory_pbm = np.asarray(trajectory_pbm)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Feasible regions: unit balls
    ball_centers = [(-2, 0), (2, 0)]
    radius = np.sqrt(0.99)
    labels = [r"$\mathbb{E}[g_1(x,y,\xi)] \leq 0 $", r"$\mathbb{E}[g_2(x,y,\xi)] \leq 0$"]

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
            center[0], center[1],
            label,
            fontsize=12,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            zorder=5
        )

    # Trajectory
    ax.plot(
        trajectory_sgd[:, 0],
        trajectory_sgd[:, 1],
        color="red",
        linewidth=2.0,
        zorder=3,
    )

    # Trajectory
    ax.plot(
        trajectory_pbm[:, 0],
        trajectory_pbm[:, 1],
        color="black",
        linewidth=2.0,
        zorder=3,
    )

    # Emphasize x_0 and x_n
    x0 = trajectory_pbm[0]
    xn = trajectory_pbm[-1]

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
        r"$x^\text{spbm}_n$",
        xy=(xn[0], xn[1]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=12,
        zorder=5,
    )

        # Emphasize x_0 and x_n
    xn = trajectory_sgd[-1]

    ax.scatter(
        xn[0], xn[1],
        s=80,
        marker="s",
        facecolor="red",
        edgecolor="red",
        zorder=4,
    )

    ax.annotate(
        r"$x^{\text{reg}}_n$",
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
    Z = X**2 + Y**2

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
    plt.savefig("demo_balls_pbm.pdf")
    plt.show()


def balls(x, sample):
    g1 = ((x[0] - 2 + sample)**2 + x[1]**2 - 1)
    g2 = ((x[0] + 2 + sample)**2 + x[1]**2 - 1)
    # g1 = ((x[0] - 2)**2 + x[1]**2 - 1) + sample
    # g2 = ((x[0] + 2)**2 + x[1]**2 - 1) + sample
    if g1 <= g2:
        return g1
    else:
        return g2

def parabola(x):
    return x[0]**2 + x[1]**2


samples = [
    # torch.tensor([0]),
    # torch.tensor([0])
    torch.tensor([-0.1]),
    torch.tensor([0.1])
]


################## SGD ######################### 

xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
with torch.no_grad():
    xy[0] = 0
    xy[1] = 1

sgd = torch.optim.SGD([xy], lr=0.02, dampening=0.1)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    sgd,
    lr_lambda=lambda step: 0.99 ** step
)

iters = 100

param_log_sgd = []
con_log_sgd = []
dual_log_sgd = []
c_grad_log_sgd = []

for i in range(iters):
    param_log_sgd.append(
        xy.detach().numpy().copy()
    )
    
    minibatch = samples[i % 2]
    
    c = balls(xy, minibatch)

    obj = parabola(xy) + torch.norm(c)

    obj.backward()
    sgd.step()
    scheduler.step()
    sgd.zero_grad()

    con_log_sgd.append(c.detach().numpy().copy().item())



################## PBM ######################### 

xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
with torch.no_grad():
    xy[0] = 0
    xy[1] = 1

pbm = PBM([xy], m=1, lr=0.008, dual_bounds=(1e-3, 1e3), penalty_update_m='CONST', epoch_len=2, mu=0, opt_method="SGD")


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
    
    c = balls(xy, minibatch)
    
    pbm.dual_step(0, c)
    dual_log.append(pbm._dual_vars.detach().numpy().copy().item())

    obj = parabola(xy)

    pbm.step(obj)
    for gr in pbm.param_groups:
        gr['lr'] *= 0.98

    con_log.append(c.detach().numpy().copy().item())


# print(param_log)
# print(con_log)

plot_balls_trajectory(
    np.array(param_log), np.array(param_log_sgd)
)

# print(param_log)
# print(np.array(dual_log))
# print(np.array(con_log))
# print(np.array(con_log))