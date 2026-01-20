import numpy as np
import matplotlib.pyplot as plt


def plot_losses_and_constraints_stochastic(
    train_losses_list,
    train_losses_std_list,
    train_constraints_list,
    train_constraints_std_list,
    constraint_thresholds,
    test_losses_list=None,
    test_losses_std_list=None,
    test_constraints_list=None,
    test_constraints_std_list=None,
    titles=None,
    eval_points=1,
    std_multiplier=2,
    log_constraints=False,
    mode="train",  # "train" or "train_test"
    plot_time_instead_epochs=False,
    save_path="./data/figs/cifar10.pdf",
    constraints_min_max=True
):
    """
    mode:
        "train"       -> only training plots
        "train_test"  -> training + test side by side
    """

    # --- Color palette (Tableau 10) ---
    colors = [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AB",
    ]

    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    num_algos = len(train_losses_list)
    if titles is None:
        titles = [f"Algorithm {i + 1}" for i in range(num_algos)]

    constraint_thresholds = np.atleast_1d(constraint_thresholds)

    # --- Layout ---
    ncols = 1 if mode == "train" else 2
    fig, axes = plt.subplots(2, ncols, figsize=(9 * ncols, 10), sharex="col")

    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # ======================================================
    # Helper plotting functions
    # ======================================================

    def plot_loss(ax, losses_list, losses_std_list, title_suffix):
        for j, (loss, loss_std) in enumerate(zip(losses_list, losses_std_list)):
            x = np.arange(len(loss))
            color = colors[j % len(colors)]
            upper = loss + std_multiplier * loss_std
            lower = loss - std_multiplier * loss_std

            ax.plot(x, loss, lw=2.2, color=color, label=titles[j])
            ax.fill_between(x, lower, upper, color=color, alpha=0.15)

            if eval_points is not None:
                idx = (
                    np.arange(0, len(loss), eval_points)
                    if isinstance(eval_points, int)
                    else np.array(eval_points)
                )
                idx = idx[idx < len(loss)]
                ax.plot(
                    x[idx],
                    loss[idx],
                    marker_styles[j % len(marker_styles)],
                    color=color,
                    markersize=6,
                    alpha=0.8,
                )

        ax.set_title(f"Loss ({title_suffix})")
        ax.set_ylabel("Mean Loss")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)

    def plot_constraints(ax, constraints_list, constraints_std_list, title_suffix):
        for j, (constraints, constraints_std) in enumerate(
            zip(constraints_list, constraints_std_list)
        ):
            color = colors[j % len(colors)]
            constraints = np.asarray(constraints)
            constraints_std = np.asarray(constraints_std)

            x = np.arange(constraints.shape[1])

            c_min = np.min(constraints - std_multiplier * constraints_std, axis=0)
            c_max = np.max(constraints + std_multiplier * constraints_std, axis=0)

            c_min_v = np.min(constraints, axis=0)
            c_max_v = np.max(constraints, axis=0)

            # if should plot only min-max constraints
            if constraints_min_max:
                ax.plot(x, c_min_v, lw=1.8, color=color, alpha=0.5, label=titles[j])
                ax.plot(x, c_max_v, lw=1.8, color=color, alpha=0.6, label=None)
                ax.fill_between(x, c_min_v, c_max_v, color=color, alpha=0.2)
            else: 
                ax.fill_between(x, c_min, c_max, color=color, alpha=0.1)

                for idx_c, c_mean in enumerate(constraints):
                    
                    if idx_c == 0:
                        label = titles[j]
                    else: 
                        label = None
                        
                    ax.plot(x, c_mean, lw=1.8, color=color, alpha=0.3, label=label)

                    if eval_points is not None:
                        idx = (
                            np.arange(0, len(c_mean), eval_points)
                            if isinstance(eval_points, int)
                            else np.array(eval_points)
                        )
                        idx = idx[idx < len(c_mean)]
                        ax.plot(
                            x[idx],
                            c_mean[idx],
                            marker_styles[j % len(marker_styles)],
                            color=color,
                            markersize=5,
                            alpha=0.3,
                        )

        for th in constraint_thresholds:
            y = np.log(th) if log_constraints else th
            ax.axhline(y, color="red", linestyle="--", lw=1.4, label="Threshold")

        ax.set_title(f"Constraint ({title_suffix})")
        ax.set_ylabel("Log Constraint" if log_constraints else "Constraint")

        if plot_time_instead_epochs:
            ax.set_xlabel("Time (m)")
        else: 
            ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)

    # ======================================================
    # TRAIN PLOTS
    # ======================================================

    plot_loss(axes[0, 0], train_losses_list, train_losses_std_list, "Train")
    plot_constraints(
        axes[1, 0],
        train_constraints_list,
        train_constraints_std_list,
        "Train",
    )

    # ======================================================
    # TEST PLOTS
    # ======================================================

    if mode == "train_test":
        plot_loss(axes[0, 1], test_losses_list, test_losses_std_list, "Test")
        plot_constraints(
            axes[1, 1],
            test_constraints_list,
            test_constraints_std_list,
            "Test",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':

    # define the torch seed here
    n_epochs = 2
    n_constraints = 90
    threshold = 0.1
    device = "cpu"

    # define seeds
    seeds = [1, 2, 3]

    # log path file
    log_path = "./data/logs/log_benchmark_stochastic_cifar_bench.npz"

    # load data
    dataloader_train, dataloader_test, features_train = load_data()

    # resave to empty file
    np.savez(
    log_path,
        losses=[],
        constraints=[],
        losses_std=[],
        constraints_std=[],
        losses_t=[],
        constraints_t=[],
        losses_std_t=[],
        constraints_std_t=[],
        times=[]
    )

    # benchmark adam
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, adam)
    print('ADAM DONE!!!')

    # benchmark ssw
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, ssw)
    print('SSW DONE!!!')

    # # benchmark sslalm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, sslalm)
    print('SSLALM DONE!!!')

    # # benchmark pbm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, pbm)
    print('PBM DONE!!!')

    # PLOT 
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])
    losses_t = list(np.load(log_path)["losses_t"])
    constraints_t = list(np.load(log_path)["constraints_t"])
    losses_std_t = list(np.load(log_path)["losses_std_t"])
    constraints_std_t = list(np.load(log_path)["constraints_std_t"])

    print('times:', list(np.load(log_path)["times"]))   

    plot_losses_and_constraints_stochastic(
        losses,
        losses_std,
        constraints,
        constraints_std,
        [threshold],
        test_losses_list=losses_t,
        test_losses_std_list=losses_std_t,
        test_constraints_list=constraints_t,
        test_constraints_std_list=constraints_std_t,
        titles=[
            "Unconstrained Adam",
            "SSW",
            "SSLALM",
            "SPBM"
        ],
        log_constraints=False,
        std_multiplier=1,
        mode='train_test', # change this to 'train', to ignore the test=
        plot_time_instead_epochs=False,
        save_path="./data/figs/cifar10_bench.pdf"
    )