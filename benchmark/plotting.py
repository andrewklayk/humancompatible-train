from matplotlib import pyplot as plt
import numpy as np


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
    times=[], # second per epoch
    plot_time_instead_epochs=False,
    save_path=None,
    separate_constraints = False,
    abs_constraints=False
):
    """
    mode:
        "train"       -> only training plots
        "train_test"  -> training + test side by side
    """

#     # --- Color palette (Tableau 10) ---
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
    nrows = 1 + train_constraints_list[0].shape[0] if separate_constraints else 2

    join_bottom_plot = not test_constraints_list and mode == "train_test"

    if join_bottom_plot:
        fig = plt.figure(figsize=(9 * ncols, 10))
        axes = []
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2, sharey = ax1)
        ax3 = fig.add_subplot(2, 1, 2)

        axes = [ax1, ax2, ax3]
    else:
        fig, axes = plt.subplots(2, ncols, figsize=(9 * ncols, 10), sharex="col", sharey="row")

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

            if plot_time_instead_epochs:
                x *= round(times[j])

            # ax.plot(x, loss, lw=2.2, color=color, label=titles[j] + f"; TPE: {minutes}m:{seconds}s")
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

            print(np.array(constraints).shape)
            c_max = np.max(constraints, axis=0)
            c_max_std = np.std(c_max)

            c_lower = c_max - std_multiplier * c_max_std
            c_upper = c_max + std_multiplier * c_max_std
            ax.fill_between(x, c_lower, c_upper, color=color, alpha=0.1)

            if plot_time_instead_epochs:
                x *= round(times[j])

            label = titles[j]
            ax.plot(x, c_max, lw=1.8, color=color, alpha=0.3, label=label)

            if eval_points is not None:
                idx = (
                    np.arange(0, len(c_max), eval_points)
                    if isinstance(eval_points, int)
                    else np.array(eval_points)
                )
                idx = idx[idx < len(c_max)]
                ax.plot(
                    x[idx],
                    c_max[idx],
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

    plot_loss(
        axes[0] if join_bottom_plot else axes[0, 0],
        train_losses_list,
        train_losses_std_list,
        "Train"
    )
    plot_constraints(
        axes[2] if join_bottom_plot else axes[1, 0],
        train_constraints_list,
        train_constraints_std_list,
        "Train",
    )

    # ======================================================
    # TEST PLOTS
    # ======================================================

    if mode == "train_test":
        plot_loss(
            axes[1] if join_bottom_plot else axes[0, 1],
            test_losses_list,
            test_losses_std_list,
            "Test"
        )
        if join_bottom_plot:
            print(axes[0].get_yticks())
            print(axes[1].get_yticks())
        if test_constraints_list:
            plot_constraints(
                axes[1, 1],
                test_constraints_list,
                test_constraints_std_list,
                "Test",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)



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
    times=None,  # seconds per epoch or list of per-epoch times
    plot_time_instead_epochs=False,
    save_path=None,
    separate_constraints=False,
    abs_constraints=False,
):
    """
    Clean, modular rewrite.

    Parameters
    ----------
    mode : str
        "train" or "train_test"
    separate_constraints : bool
        If True, plots each constraint in separate row.
        If False, plots max constraint.
    """

    # ------------------------------------------------------------------
    # Basic setup
    # ------------------------------------------------------------------

    num_algos = len(train_losses_list)

    if titles is None:
        titles = [f"Algorithm {i+1}" for i in range(num_algos)]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    constraint_thresholds = np.atleast_1d(constraint_thresholds)
    times = times or [None] * num_algos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def make_x(n, algo_idx):
        if not plot_time_instead_epochs:
            return np.arange(n)

        t = times[algo_idx]
        if t is None:
            return np.arange(n)

        if np.isscalar(t):
            return np.cumsum(np.full(n, t))
        else:
            return np.cumsum(t[:n])

    def get_eval_idx(n):
        if eval_points is None:
            return None
        if isinstance(eval_points, int):
            return np.arange(0, n, eval_points)
        return np.asarray(eval_points)

    def plot_curve(ax, x, y, y_std, color, label, marker=None, alpha_line=1.0):
        ax.plot(x, y, lw=2, color=color, label=label, alpha=alpha_line)
        ax.fill_between(
            x,
            y - std_multiplier * y_std,
            y + std_multiplier * y_std,
            color=color,
            alpha=0.15,
        )

        idx = get_eval_idx(len(y))
        if idx is not None:
            idx = idx[idx < len(y)]
            ax.plot(
                x[idx],
                y[idx],
                linestyle="None",
                marker=marker,
                color=color,
                markersize=5,
                alpha=0.8,
            )

    def compute_max_constraint(c, c_std):
        idx = np.argmax(c, axis=0)
        c_max = c[idx, np.arange(c.shape[1])]
        c_max_std = c_std[idx, np.arange(c.shape[1])]
        return c_max, c_max_std

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    if mode == "train":
        ncols = 1
    else:
        ncols = 2

    if separate_constraints:
        n_constraints = train_constraints_list[0].shape[0]
        nrows = 1 + n_constraints
    else:
        nrows = 2

    # Special handling: when train_test mode but no test constraints, span full width
    constraint_spans_full_width = (
        mode == "train" and test_constraints_list is None
    )

    if constraint_spans_full_width:
        # Create figure manually for full-width constraint
        fig = plt.figure(figsize=(16, 4.5 * nrows))
        ax_loss_train = fig.add_subplot(nrows, 2, 1)
        ax_loss_test = fig.add_subplot(nrows, 2, 2)
        axes_loss = [[ax_loss_train, ax_loss_test]]
        axes_constraint = []
        for k in range(n_constraints if separate_constraints else 1):
            ax_c = fig.add_subplot(nrows, 1, 2 + k)
            axes_constraint.append([ax_c])
    else:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8 * ncols, 4.5 * nrows),
            sharex="col",
            sharey="row",
        )
        axes = np.atleast_2d(axes)
    # ------------------------------------------------------------------
    # LOSS PLOTTING
    # ------------------------------------------------------------------

    def plot_loss_panel(ax, losses_list, losses_std_list, title):
        for j, (loss, loss_std) in enumerate(zip(losses_list, losses_std_list)):
            x = make_x(len(loss), j)
            plot_curve(
                ax,
                x,
                loss,
                loss_std,
                color=colors[j % len(colors)],
                label=titles[j],
                marker=markers[j % len(markers)],
            )

        ax.set_title(f"Loss ({title})")
        ax.set_ylabel("Mean Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        # ax.set_ylim(bottom=0.3, top=0.7)
        ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # CONSTRAINT PLOTTING
    # ------------------------------------------------------------------

    def plot_constraint_panel(ax, constraints_list, constraints_std_list, title, constraint_idx=None):
        for j, (c, c_std) in enumerate(zip(constraints_list, constraints_std_list)):
            if abs_constraints:
                c = np.abs(c)

            if separate_constraints:
                y = c[constraint_idx]
                y_std = c_std[constraint_idx]
                label = f"{titles[j]}"
            else:
                y, y_std = compute_max_constraint(c, c_std)
                label = titles[j]

            if log_constraints:
                y = np.log(np.clip(y, 1e-12, None))

            x = make_x(len(y), j)

            plot_curve(
                ax,
                x,
                y,
                y_std,
                color=colors[j % len(colors)],
                label=label,
                marker=markers[j % len(markers)],
                alpha_line=0.9,
            )

        for i, th in enumerate(constraint_thresholds):
            th_val = np.log(th) if log_constraints else th
            ax.axhline(
                th_val,
                color="red",
                linestyle="--",
                lw=1.4,
                label="Threshold" if i == 0 else None,
            )

        ax.set_title(f"Constraint ({title})")
        ax.set_ylabel("Log Constraint" if log_constraints else "Constraint")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    if constraint_spans_full_width:
        plot_loss_panel(axes_loss[0][0], train_losses_list, train_losses_std_list, "Train")
        plot_loss_panel(axes_loss[0][1], test_losses_list, test_losses_std_list, "Test")
        
        # Plot train constraints spanning full width
        if separate_constraints:
            for k in range(n_constraints):
                plot_constraint_panel(
                    axes_constraint[k][0],
                    train_constraints_list,
                    train_constraints_std_list,
                    "Train",
                    constraint_idx=k,
                )
                axes_constraint[k][0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
        else:
            plot_constraint_panel(
                axes_constraint[0][0],
                train_constraints_list,
                train_constraints_std_list,
                "Train",
            )
            axes_constraint[0][0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
    else:
        # Normal grid-based layout
        axes = np.atleast_2d(axes)
        plot_loss_panel(axes[0, 0], train_losses_list, train_losses_std_list, "Train")
        
        if mode == "train_test":
            plot_loss_panel(axes[0, 1], test_losses_list, test_losses_std_list, "Test")

        # Plot constraints
        if mode == "train":
            # When only train mode, constraint spans full width (or just uses column 0)
            if separate_constraints:
                for k in range(n_constraints):
                    plot_constraint_panel(
                        axes[1 + k, 0],
                        train_constraints_list,
                        train_constraints_std_list,
                        "Train",
                        constraint_idx=k,
                    )
                    axes[1 + k, 0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
            else:
                plot_constraint_panel(
                    axes[1, 0],
                    train_constraints_list,
                    train_constraints_std_list,
                    "Train",
                )
                axes[1, 0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
        
        elif mode == "train_test":
            # Both train and test constraints provided
            if separate_constraints:
                for k in range(n_constraints):
                    plot_constraint_panel(
                        axes[1 + k, 0],
                        train_constraints_list,
                        train_constraints_std_list,
                        "Train",
                        constraint_idx=k,
                    )
                    axes[1 + k, 0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
                    plot_constraint_panel(
                        axes[1 + k, 1],
                        test_constraints_list,
                        test_constraints_std_list,
                        "Test",
                        constraint_idx=k,
                    )
                    axes[1 + k, 1].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
            else:
                plot_constraint_panel(
                    axes[1, 0],
                    train_constraints_list,
                    train_constraints_std_list,
                    "Train",
                )
                axes[1, 0].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
                plot_constraint_panel(
                    axes[1, 1],
                    test_constraints_list,
                    test_constraints_std_list,
                    "Test",
                )
                axes[1, 1].set_xlabel("Time" if plot_time_instead_epochs else "Epoch")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


import numpy as np
import matplotlib.pyplot as plt


# def plot_losses_and_constraints_stochastic(
#     train_losses_list,
#     train_losses_std_list,
#     train_constraints_list,
#     train_constraints_std_list,
#     constraint_thresholds,
#     test_losses_list=None,
#     test_losses_std_list=None,
#     test_constraints_list=None,
#     test_constraints_std_list=None,
#     titles=None,
#     std_multiplier=2,
#     log_constraints=False,
#     plot_time_instead_epochs=False,
#     plot_max_constraint=False,
#     times=None,
#     save_path=None,
#     combine_algos=False,
# ):
#     """
#     Layouts:
    
#     combine_algos=False (default):
#         Columns  -> algorithms
#         Row 0    -> Loss (train + test together)
#         Row i>0  -> Constraint i (train + test together)
    
#     combine_algos=True:
#         Columns  -> train / test
#         Row 0    -> Loss
#         Row i>0  -> Constraint i
#         All algorithms on same plot with different colors.
#     """

#     num_algos = len(train_losses_list)
#     times = times or [None] * num_algos
#     constraint_thresholds = np.atleast_1d(constraint_thresholds)

#     # ----------------------------------------------------------
#     # Determine number of constraint rows
#     # ----------------------------------------------------------

#     if plot_max_constraint:
#         n_constraints = 1
#     else:
#         n_constraints = train_constraints_list[0].shape[0]

#     if titles is None:
#         titles = [f"Algorithm {i+1}" for i in range(num_algos)]

#     nrows = 1 + n_constraints
#     ncols = 2 if combine_algos else num_algos

#     fig, axes = plt.subplots(
#         nrows,
#         ncols,
#         figsize=(5 * ncols, 3.8 * nrows) if not combine_algos else (10, 3.8 * nrows),
#         sharex="col",
#         sharey="row",  # ensures constraints share Y axis
#     )

#     axes = np.atleast_2d(axes)
    
#     # Color palette
#     # colors = plt.cm.tab10(np.linspace(0, 1, num_algos))
#     colors = [
#         "#4E79A7",
#         "#F28E2B",
#         "#E15759",
#         "#76B7B2",
#         "#59A14F",
#         "#EDC948",
#         "#B07AA1",
#         "#FF9DA7",
#         "#9C755F",
#         "#BAB0AB",
#     ]

#     # ----------------------------------------------------------
#     # Helpers
#     # ----------------------------------------------------------

#     def make_x(n, algo_idx):
#         if not plot_time_instead_epochs:
#             return np.arange(n)

#         t = times[algo_idx]
#         if t is None:
#             return np.arange(n)

#         if np.isscalar(t):
#             return np.cumsum(np.full(n, t))
#         return np.cumsum(t[:n])

#     def plot_with_std(ax, x, y, y_std, color, linestyle, label=None):
#         ax.plot(x, y, lw=2, color=color, linestyle=linestyle, label=label)
#         ax.fill_between(
#             x,
#             y - std_multiplier * y_std,
#             y + std_multiplier * y_std,
#             color=color,
#             alpha=0.15,
#         )

#     def compute_max_constraint(c, c_std):
#         """
#         Compute largest constraint at each step,
#         preserving correct std of active constraint.
#         """
#         idx = np.argmax(c, axis=0)
#         c_max = c[idx, np.arange(c.shape[1])]
#         c_max_std = c_std[idx, np.arange(c.shape[1])]
#         return c_max, c_max_std

#     # ----------------------------------------------------------
#     # Plot per algorithm (per column) OR all algos combined
#     # ----------------------------------------------------------

#     if combine_algos:
#         # Layout: columns are train/test, rows are loss/constraints
#         # All algorithms on same plot with different colors
        
#         # TRAIN column (col 0)
#         ax_loss_train = axes[0, 0]
#         for j in range(num_algos):
#             x_train = make_x(len(train_losses_list[j]), j)
#             plot_with_std(
#                 ax_loss_train,
#                 x_train,
#                 train_losses_list[j],
#                 train_losses_std_list[j],
#                 color=colors[j],
#                 linestyle="-",
#                 label=titles[j],
#             )
        
#         ax_loss_train.set_title("Loss (Train)")
#         ax_loss_train.set_ylabel("Loss")
#         ax_loss_train.grid(True, linestyle="-", alpha=0.4)
#         ax_loss_train.legend(fontsize=9, loc='best')
        
#         # TEST column (col 1)
#         if test_losses_list is not None:
#             ax_loss_test = axes[0, 1]
#             for j in range(num_algos):
#                 x_test = make_x(len(test_losses_list[j]), j)
#                 plot_with_std(
#                     ax_loss_test,
#                     x_test,
#                     test_losses_list[j],
#                     test_losses_std_list[j],
#                     color=colors[j],
#                     linestyle="-",
#                     label=titles[j],
#                 )
            
#             ax_loss_test.set_title("Loss (Test)")
#             ax_loss_test.set_ylabel("Loss")
#             ax_loss_test.grid(True, linestyle="-", alpha=0.4)
#             ax_loss_test.legend(fontsize=9, loc='best')
        
#         # Constraints
#         for k in range(n_constraints):
#             # TRAIN constraints (col 0)
#             ax_c_train = axes[1 + k, 0]
#             for j in range(num_algos):
#                 c_train = train_constraints_list[j]
#                 c_train_std = train_constraints_std_list[j]
                
#                 if plot_max_constraint:
#                     c_train, c_train_std = compute_max_constraint(
#                         c_train, c_train_std
#                     )
#                 else:
#                     c_train = c_train[k]
#                     c_train_std = c_train_std[k]
                
#                 if log_constraints:
#                     c_train = np.log(np.clip(c_train, 1e-12, None))
                
#                 x_train = make_x(len(c_train), j)
                
#                 plot_with_std(
#                     ax_c_train,
#                     x_train,
#                     c_train,
#                     c_train_std,
#                     color=colors[j],
#                     linestyle="-",
#                     label=titles[j] if k == 0 else None,
#                 )
            
#             # Threshold
#             if plot_max_constraint:
#                 th = np.max(constraint_thresholds)
#             else:
#                 th = constraint_thresholds[min(k, len(constraint_thresholds) - 1)]
            
#             th_val = np.log(th) if log_constraints else th
#             ax_c_train.axhline(th_val, color="black", linestyle=":", lw=1.5)
            
#             ylabel = "Max Constraint" if plot_max_constraint else f"C{k+1}"
#             ax_c_train.set_ylabel(ylabel)
#             ax_c_train.grid(True, linestyle="--", alpha=0.4)
#             ax_c_train.set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
            
#             # TEST constraints (col 1)
#             if test_constraints_list is not None:
#                 ax_c_test = axes[1 + k, 1]
#                 for j in range(num_algos):
#                     c_test = test_constraints_list[j]
#                     c_test_std = test_constraints_std_list[j]
                    
#                     if plot_max_constraint:
#                         c_test, c_test_std = compute_max_constraint(
#                             c_test, c_test_std
#                         )
#                     else:
#                         c_test = c_test[k]
#                         c_test_std = c_test_std[k]
                    
#                     if log_constraints:
#                         c_test = np.log(np.clip(c_test, 1e-12, None))
                    
#                     x_test = make_x(len(c_test), j)
                    
#                     plot_with_std(
#                         ax_c_test,
#                         x_test,
#                         c_test,
#                         c_test_std,
#                         color=colors[j],
#                         linestyle="-",
#                         label=titles[j] if k == 0 else None,
#                     )
                
#                 ax_c_test.axhline(th_val, color="black", linestyle=":", lw=1.5)
#                 ax_c_test.set_ylabel(ylabel)
#                 ax_c_test.grid(True, linestyle="--", alpha=0.4)
#                 ax_c_test.set_xlabel("Time" if plot_time_instead_epochs else "Epoch")
    
#     else:
#         # Default layout: columns are algorithms, train/test on same plot
#         for j in range(num_algos):

#             # =========================
#             # LOSS
#             # =========================

#             ax_loss = axes[0, j]

#             x_train = make_x(len(train_losses_list[j]), j)
#             plot_with_std(
#                 ax_loss,
#                 x_train,
#                 train_losses_list[j],
#                 train_losses_std_list[j],
#                 color="tab:blue",
#                 linestyle="-",
#                 label="Train",
#             )

#             if test_losses_list is not None:
#                 x_test = make_x(len(test_losses_list[j]), j)
#                 plot_with_std(
#                     ax_loss,
#                     x_test,
#                     test_losses_list[j],
#                     test_losses_std_list[j],
#                     color="tab:orange",
#                     linestyle="-",
#                     label="Test",
#                 )

#             ax_loss.set_title(titles[j])
#             ax_loss.set_ylabel("Loss")
#             ax_loss.grid(True, linestyle="-", alpha=0.4)

#             if j == 0:
#                 ax_loss.legend()

#             # =========================
#             # CONSTRAINTS
#             # =========================

#             for k in range(n_constraints):

#                 ax_c = axes[1 + k, j]

#                 # ----- TRAIN -----

#                 c_train = train_constraints_list[j]
#                 c_train_std = train_constraints_std_list[j]

#                 if plot_max_constraint:
#                     c_train, c_train_std = compute_max_constraint(
#                         c_train, c_train_std
#                     )
#                 else:
#                     c_train = c_train[k]
#                     c_train_std = c_train_std[k]

#                 if log_constraints:
#                     c_train = np.log(np.clip(c_train, 1e-12, None))

#                 x_train = make_x(len(c_train), j)

#                 plot_with_std(
#                     ax_c,
#                     x_train,
#                     c_train,
#                     c_train_std,
#                     color="tab:blue",
#                     linestyle="-",
#                     label="Train" if k == 0 else None,
#                 )

#                 # ----- TEST -----

#                 if test_constraints_list is not None:
#                     c_test = test_constraints_list[j]
#                     c_test_std = test_constraints_std_list[j]

#                     if plot_max_constraint:
#                         c_test, c_test_std = compute_max_constraint(
#                             c_test, c_test_std
#                         )
#                     else:
#                         c_test = c_test[k]
#                         c_test_std = c_test_std[k]

#                     if log_constraints:
#                         c_test = np.log(np.clip(c_test, 1e-12, None))

#                     x_test = make_x(len(c_test), j)

#                     plot_with_std(
#                         ax_c,
#                         x_test,
#                         c_test,
#                         c_test_std,
#                         color="tab:orange",
#                         linestyle="-",
#                         label="Test" if k == 0 else None,
#                     )

#                 # Threshold (draw once per row)
#                 if plot_max_constraint:
#                     th = np.max(constraint_thresholds)
#                 else:
#                     th = constraint_thresholds[min(k, len(constraint_thresholds) - 1)]

#                 th_val = np.log(th) if log_constraints else th

#                 ax_c.axhline(
#                     th_val,
#                     color="black",
#                     linestyle=":",
#                     lw=1.5,
#                 )

#                 if j == 0:
#                     ylabel = "Max Constraint" if plot_max_constraint else f"C{k+1}"
#                     ax_c.set_ylabel(ylabel)

#                 ax_c.grid(True, linestyle="--", alpha=0.4)

#                 if k == n_constraints - 1:
#                     ax_c.set_xlabel("Time" if plot_time_instead_epochs else "Epoch")

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#     plt.show()