from sched import scheduler

from matplotlib.patches import Circle
import torch
from humancompatible.train.optim.PBM import PBM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM
torch.manual_seed(1)
np.random.seed(1)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_balls_trajectory(trajectories, names):
    fig, ax = plt.subplots(figsize=(24, 16))

    # Feasible regions: unit balls
    ball_centers = [(-2, 0), (2, 0)]
    radius = np.sqrt(0.99)
    labels = [r"$\mathbb{E}[g(x,\xi)] \leq 0$", r"$\mathbb{E}[g(x,\xi)] \leq 0$"]

    # Heatmap for x^2 + y^2
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Draw balls
    for center, label in zip(ball_centers, labels):
        ball = Circle(
            center,
            radius=radius,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=6,  # Thickest practical line
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(ball)
        ax.text(
            center[0], center[1],
            label,
            fontsize=40,  # Very large font
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            zorder=5
        )

    # Plot trajectories
    for i, traj in enumerate(trajectories):
        traj = np.asarray(traj)
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=6,  # Thickest practical line
            zorder=3,
            alpha=1.0,
            color= 'c' if i == 0 else ('orange' if i == 1 else ('tab:green' if i == 2 else 'red'))
        )

        # Emphasize x_0 and x_n
        x0 = traj[0]
        xn = traj[-1]

        ax.scatter(
            x0[0], x0[1],
            s=400,  # Very large marker
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=4,  # Thick edge
            zorder=4,
        )
        ax.scatter(
            xn[0], xn[1],
            s=300,  # Very large marker
            marker="s",
            facecolor="black",
            edgecolor="black",
            zorder=4,
        )

        # Labels for x_0 and x_n
        x_n = [
            r"$x_{\rho=0}^n$",
            r"$x_{\rho=1}^n$",
            r"$x_{\rho=2.5}^n$",
            r"$x_{SPBM}^n$",
        ]
        ax.annotate(
            r"$x^0$",
            xy=(x0[0], x0[1]),
            xytext=(15, 15),  # Adjusted offset
            textcoords="offset points",
            fontsize=40,  # Very large font
            zorder=5,
        )
        
        ax.annotate(
            x_n[i],
            xy=(xn[0], xn[1]),
            xytext=(-25, -50) if i == 0 else ((10, 40) if i == 1 else ((30, -70) if i == 2 else (-80, 150))),
            textcoords="offset points",
            fontsize=55,  # Very large font
            zorder=5,
            color= 'c' if i == 0 else ('orange' if i == 1 else ('tab:green' if i == 2 else 'red'))
        )

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$", fontsize=40)  # Very large font
    ax.set_ylabel(r"$x_2$", fontsize=40)  # Very large font
    ax.grid(True, linestyle=":", linewidth=2, alpha=0.7)  # Thicker grid
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.8, 1.8)
    ax.tick_params(axis='both', which='major', labelsize=32)  # Large tick fonts

    # Contour/heatmap
    contour = ax.contourf(
        X, Y, Z,
        levels=100,
        cmap='viridis',
        alpha=0.6,  # More opaque
        zorder=0
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)  # Wider colorbar
    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label(r"$||x||^2$", fontsize=40)  # Very large font
    cbar.ax.tick_params(labelsize=32)  # Large colorbar tick fonts

    # Save with highest practical DPI
    fig.savefig(
        "./demo_balls_pbm.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=100  # Highest practical DPI
    )



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

def run_sgd(rho: float):

    xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
    with torch.no_grad():
        xy[0] = 0
        xy[1] = 1

    sgd = torch.optim.SGD([xy], lr=0.05, dampening=0.1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sgd,
        lr_lambda=lambda step: 0.99 ** step
    )

    param_log_sgd = []
    con_log_sgd = []

    for i in range(200):
        param_log_sgd.append(
            xy.detach().numpy().copy()
        )

        r = np.random.uniform()
        minibatch = samples[0 if r > 0.5 else 1]
            
        c = balls(xy, minibatch)

        obj = parabola(xy) + rho * torch.square(torch.norm(c, p=2))

        obj.backward()
        sgd.step()
        scheduler.step()
        sgd.zero_grad()
        for gr in sgd.param_groups:
            gr['lr'] *= 0.97

        con_log_sgd.append(c.detach().numpy().copy().item())
    
    return param_log_sgd, con_log_sgd


sgd_param_logs, sgd_con_logs = [], []

rhos = [0.1,1,2]

for rho in rhos:
    param_log_sgd, con_log_sgd = run_sgd(rho)
    sgd_param_logs.append(param_log_sgd)
    sgd_con_logs.append(con_log_sgd)


################## PBM ######################### 

xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
with torch.no_grad():
    xy[0] = 0
    xy[1] = 1

# Define data and optimizers
optimizer = MoreauEnvelope(torch.optim.SGD([xy], lr=0.01), mu=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: 0.99 ** step
)

dual = PBM(
    m=1,
    penalty_update='const',
    pbf = 'quadratic_logarithmic',
    gamma=0.0,
    init_duals=0.1,
    init_penalties=1.,
    penalty_range=(0.5, 1.),
    penalty_mult=0.99,
    dual_range=(0.1, 10.)
)

iters = 200

param_log = []
con_log = []
dual_log = []
c_grad_log = []

for i in range(iters):
    param_log.append(
        xy.detach().numpy().copy()
    )
    # print(xy)
    
    r = np.random.uniform()
    minibatch = samples[
        1 if r > 0.5 else 0
        ]
    
    c = balls(xy, minibatch)
    obj = parabola(xy)
    
    # compute the lagrangian value
    lagrangian = dual.forward_update(obj, c)
    lagrangian.backward()
    optimizer.step()
    optimizer.zero_grad()

    scheduler.step()

    dual_log.append(dual.duals.detach().numpy().copy().item())
    con_log.append(c.detach().numpy().copy().item())


# print(param_log)
# print(con_log)

trajectories = sgd_param_logs
trajectories.append(param_log)

plot_balls_trajectory(
    trajectories,
    [r"$\rho= $" + str(rho) for rho in rhos] + ['spbm']
)

# print(param_log)
# print(np.array(dual_log))
# print(np.array(con_log))
# print(np.array(con_log))  