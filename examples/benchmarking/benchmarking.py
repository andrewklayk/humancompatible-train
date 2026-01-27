import torch
import torch.nn as nn
import torch.optim as optim
from fairret.statistic import Accuracy
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import copy
import numpy as np
from humancompatible.train.optim.ssw import SSG
from humancompatible.train.optim.PBM import PBM
from humancompatible.train.optim.ssl_alm_adam import SSLALM_Adam
import timeit

"""
Helper functions for benchmarking notebook
"""

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
    abs_constraints=False
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

    join_bottom_plot = not test_constraints_list and mode == "train_test"

    if join_bottom_plot:
        fig, axes = plt.subplot_mosaic([[0, 1], [2, 2]], figsize=(9 * ncols, 10))
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


            # c_min = np.min(constraints - std_multiplier * constraints_std, axis=0)
            # c_max = np.max(constraints + std_multiplier * constraints_std, axis=0)
            # ax.fill_between(x, c_min, c_max, color=color, alpha=0.1)

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
                

            # for c_idx, c_mean in enumerate(constraints):

                # if plot_time_instead_epochs:
                #     x *= round(times[j])

                # if c_idx == 0:
                #     label = titles[j]
                # else: 
                #     label = None

                # ax.plot(x, c_mean, lw=1.8, color=color, alpha=0.3, label=label)

                # if eval_points is not None:
                #     idx = (
                #         np.arange(0, len(c_mean), eval_points)
                #         if isinstance(eval_points, int)
                #         else np.array(eval_points)
                #     )
                #     idx = idx[idx < len(c_mean)]
                #     ax.plot(
                #         x[idx],
                #         c_mean[idx],
                #         marker_styles[j % len(marker_styles)],
                #         color=color,
                #         markersize=5,
                #         alpha=0.3,
                #     )

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
            axes[0].set_yticks(axes[1].get_yticks())
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


def cifar_train(network_achitecture, n_epochs, seed_n, trainloader, loss_per_class_f, test_network_f, device, classes_arr, fair_crit_bound, print_n, method='unconstrained',
                        model_params=None, init_weights=None):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # create the network
    net = network_achitecture(init_weights, num_classes=len(classes_arr))
    net.to(device)

    # define the loss function and the 
    criterion = nn.CrossEntropyLoss(reduction="none")

    # define number of constraints of the demographic parity
    num_constraints = len(classes_arr) * (len(classes_arr)-1)

    # create a mask for later
    N = len(classes_arr)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)

    # set the model parameters based on the optimizer
    if method == "unconstrained":
        lr = model_params['lr']
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif method == 'ssw':
        lr = model_params['lr']
        dual_lr = model_params['dual_lr']
        optimizer = SSG(params=net.parameters(), m=1, lr=lr, dual_lr=dual_lr)
    elif method == 'ssl-alm':
        lr = model_params['lr']
        dual_lr = model_params['dual_lr']
        mu = model_params['mu']
        rho = model_params['rho']
        optimizer = SSLALM_Adam(
            params=net.parameters(),
            m=num_constraints,  # number of constraints - one in our case
            lr=lr,  # primal variable lr
            dual_lr=dual_lr,  # lr of a dual ALM variable
            dual_bound=5,
            rho=rho,  # rho penalty in ALM parameter
            mu=mu,  # smoothing parameter
            device=device,
        )
    elif method == "pbm":
        lr = model_params['lr']
        dual_beta = model_params['dual_beta']
        mu = model_params['mu']
        penalty = model_params['penalty']
        init_dual = model_params['init_dual']
        warm_start = model_params['warm_start']
        penalty_update_m = model_params['p_update']
        optimizer = PBM(params=net.parameters(), m=num_constraints, lr=lr, dual_beta=dual_beta, mu=mu, 
                epoch_len=len(trainloader), init_dual=init_dual, penalty_update_m=penalty_update_m, p_lb=0.1, warm_start=warm_start,
                barrier=penalty, device=device)
    else: 
        raise ValueError("No such method available!")

    # alloc arrays for plotting
    S_loss_log_plotting = []  # mean
    S_c_log_plotting = []  # mean
    S_loss_std_log_plotting = []  # std
    S_c_std_log_plotting = []  # std

    test_S_loss_log_plotting = []  # mean
    test_S_c_log_plotting = []  # mean
    test_S_loss_std_log_plotting = []  # std
    test_S_c_std_log_plotting = []  # std

    # save the accuracy per class
    accuracy_plotting = []
    accuracy_per_class_plotting = []
    accuracy_plotting_t = []
    accuracy_per_class_plotting_t = []

    time = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # predictions + predictions per group stats
        total_pred_epoch = {classname: 0 for classname in classes_arr}
        correct_pred_epoch = {classname: 0 for classname in classes_arr}
        total_epoch = 0
        correct_epoch = 0

        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        for i, data in enumerate(trainloader, 0):

            ############################ FORWARD ###########

            # get the inputs; data is a list of [inputs, labels]
            inputs, sens, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            time_start = timeit.default_timer()
            outputs = net(inputs)
            time += timeit.default_timer() - time_start
            
            _, predicted = torch.max(outputs, 1) # compute the classes predictions

            ############################ CONSTRAINTS + STATISTICS ###########
            
            # compute loss
            time_start = timeit.default_timer()                        
            loss_per_sample = criterion(outputs, labels)
            loss = loss_per_sample.mean()
            time += timeit.default_timer() - time_start

            # loss per class
            time_start = timeit.default_timer()
            loss_per_class = loss_per_class_f(loss_per_sample, labels)
            if method != "unconstrained":
                time += timeit.default_timer() - time_start

            ## CONSTRAINT COMPUTATION
            if method != 'unconstrained':
                
                time_start = timeit.default_timer()
                
                diff = loss_per_class.unsqueeze(1) - loss_per_class.unsqueeze(0)
                constr = (diff - fair_crit_bound)[mask]   # shape: (N*(N-1),)
                if method == 'ssl-alm':
                    constr = torch.max(constr, torch.zeros_like(constr, device=device))
                elif method == 'ssw':
                    constr = torch.max(constr)
                
                time += timeit.default_timer() - time_start
                
                c_log.append(diff[mask].detach().cpu().numpy())

            else: # unsconstrained - just log the constraint 
                with torch.no_grad():
                    diff = loss_per_class.unsqueeze(1) - loss_per_class.unsqueeze(0)
                    c_log.append(diff[mask].cpu().numpy())

            ## PARAM UPDATE
            time_start = timeit.default_timer()

            if method == "unconstrained":
                loss.backward()
                optimizer.step()
            
            elif method == 'ssw':
                if constr > 0: # don't need constraint grad if not using it
                    constr.backward()
                    optimizer.dual_step(0)
                    optimizer.zero_grad()
                else:        
                    loss.backward()
                optimizer.step(constr)
                optimizer.zero_grad()

            elif method == 'pbm' or method == 'ssl-alm':
                optimizer.dual_steps(constr)
                optimizer.step(loss)
            
            time += timeit.default_timer() - time_start

            # save the logs
            loss_log.append(loss.detach().cpu().numpy())
            
            if method == 'pbm' or method == 'ssl-alm':
                duals_log.append(optimizer._dual_vars.detach().cpu())

            ############################ PRINT ###########

            # compute the accuracy overall
            total_epoch += labels.size(0)
            correct_epoch += (predicted == labels).sum().item()

            # compute the constraints - accuracy among all groups
            for idx, zipped in enumerate(zip(labels, predicted)):
                label, prediction = zipped
                if label == prediction:
                    correct_pred_epoch[classes_arr[label]] += 1
                    
                # save the number of samples per that group in the batch
                total_pred_epoch[classes_arr[label]] += 1   

            # print the statistics
            if i % print_n == print_n-1:    # print every 2000 mini-batches

                S_loss_log_plotting.append(np.mean(loss_log))
                S_c_log_plotting.append(np.mean(c_log, axis=0))
                S_loss_std_log_plotting.append(np.std(loss_log, axis=0))
                S_c_std_log_plotting.append(np.std(c_log, axis=0))

                # test the network after each epoch
                losses_test, c_test, overall_accuracy_t, acc_pergroup_t = test_network_f(net)

                # compute the accuracy per class    
                for classname, correct_count in correct_pred_epoch.items():
                    accuracy = float(correct_count) / total_pred_epoch[classname]
                    correct_pred_epoch[classname] = accuracy

                test_S_loss_log_plotting.append(np.mean(losses_test))
                test_S_c_log_plotting.append(np.mean(c_test, axis=0))
                test_S_loss_std_log_plotting.append(np.std(losses_test, axis=0))
                test_S_c_std_log_plotting.append(np.std(c_test, axis=0))

                accuracy_plotting += [correct_epoch/total_epoch]
                accuracy_per_class_plotting += [ copy.deepcopy(correct_pred_epoch) ]
                accuracy_plotting_t += [overall_accuracy_t]
                accuracy_per_class_plotting_t += [ copy.deepcopy(acc_pergroup_t) ]

                # print of the dual variables
                if method != 'ssw' and method != 'unconstrained':
                    str_dual_print = f"dual: {np.mean(duals_log, axis=0)}"
                else: 
                    str_dual_print = ""
                
                print(
                    f"Step: {epoch}, {i + 1:5d}, "
                    f"loss ({np.mean(loss_log):.4f}/{np.mean(losses_test):.4f}):"
                    f"constraints (train/test): ({np.max(np.mean(c_log, axis=0))}/{np.max(np.mean(c_test, axis=0))}), "
                    f"Accuracy (train/test): {correct_epoch/total_epoch}/{overall_accuracy_t}\n"
                    f"Accuracy Per group (train): {correct_pred_epoch}\n"
                    f"{str_dual_print}"
                )

                # restart the stats        
                total_pred_epoch = {classname: 0 for classname in classes_arr}
                correct_pred_epoch = {classname: 0 for classname in classes_arr}
                total_epoch = 0
                correct_epoch = 0
                
    print('Finished Training')

    # return the statistics
    return S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
            accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t, time