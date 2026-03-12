from matplotlib.patches import Circle
import torch
from humancompatible.train.optim.PBM import PBM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn import Sequential
from torch.optim import Adam, SGD
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM

def f(xy):
    return -xy[0] - xy[1]

def g1(xy):
    return 0.1*( (xy[0]-1)**2 + (xy[1]+1)**2 - 0.1 ) * ( (xy[0]+3)**2 + (xy[1]+3)**2 - 1.0 )

def g2(xy):
    return 0.1*( (xy[0]+1)**2 + (xy[1]-1)**2 - 0.1 ) * ( (xy[0]+3)**2 + (xy[1]+3)**2 - 1.0 )

def plot_balls_trajectory(trajectories, names):
    """
    trajectory: array-like of shape (N, 2), where each row is [x, y]
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Feasible regions: unit balls
    ball_centers = [(-1, 1), (1, -1), (-3, -3)]
    radius = [np.sqrt(0.1), np.sqrt(0.1), 1]
    labels = [r"g2 <= 0", r"g1 <= 0", r"g1 <= 0 and g2 <= 0"]

    # Heatmap for x^2 + y^2
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2


    for center, radi, label in zip(ball_centers, radius, labels):
        ball = Circle(
            center,
            radius=radi,
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
            fontsize=18,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            zorder=5
        )

    for i, traj in enumerate(trajectories):
        traj = np.asarray(traj)

        # Trajectory
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=2.0,
            zorder=3,
            alpha=1.0
        )

        # Emphasize x_0 and x_n
        x0 = traj[0]
        xn = traj[-1]

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
            s=60,
            marker="s",
            facecolor="black",
            edgecolor="black",
            zorder=4,
        )
        x_n = [
            r"$x^n$",
        ]
        
        ax.annotate(
            r"$x^0$",
            xy=(x0[0], x0[1]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=22,
            zorder=5,
        )
        ax.annotate(
            x_n[i],
            xy=(xn[0], xn[1]),
            # xytext=(8 if i==1 or i == 3 else -36, -12),
            xytext=(8 if i==1 or i == 3 else -45, -15),
            textcoords="offset points",
            fontsize=22,
            zorder=5,
        )

    # Formatting
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.8, 1.8)

    # contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis', alpha=0.5, zorder=0)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.08)

    # cbar = fig.colorbar(contour, cax=cax)
    # cbar.set_label(r"$x^2 + y^2$", fontsize=14)

    plt.show()
    # fig.savefig(
    #     "./demo_balls_pbm.pdf",
    #     bbox_inches="tight",
    #     pad_inches=0.05
    # )

def main():

    xy = torch.nn.Parameter(data=torch.ones(2, requires_grad=True))
    with torch.no_grad():
        xy[0] = 1
        xy[1] = 1

    # Define data and optimizers
    optimizer = MoreauEnvelope(SGD([xy], lr=0.001), mu=0.0)

    dual = PBM(m=2, mu=0.1,
        penalty_update='const',
        lr=0.95,
        penalty_range=(0.001, 100),
        init_penalties=1.,
        dual_range=(0.01, 100),
        init_duals=0.01,
        primal_update_process_length=20
    )
    iters = 100

    param_log = []
    con_log = []
    dual_log = []
    c_grad_log = []


    for i in range(iters):
        param_log.append(
            xy.detach().numpy().copy()
        )
        dual_log.append( dual.duals.detach().numpy().copy() )

        c1 = g1(xy)
        c2 = g2(xy)

        loss = f(xy)

        # Unsqueeze to make them 1D
        c1_unsqueezed = c1.unsqueeze(0)
        c2_unsqueezed = c2.unsqueeze(0)

        # Concatenate
        constraints = torch.cat((c1_unsqueezed, c2_unsqueezed))
        lagrangian = dual.forward_update(loss, constraints)

        lagrangian.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(dual_log[-1])
        print(param_log[-1])
    
    # plot 
    plot_balls_trajectory([param_log] , None  )



if __name__ == '__main__':
    main()

