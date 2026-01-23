"""
python script with cifar10 benchmarking
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch
from torch.nn import Sequential
from folktables import ACSDataSource, ACSIncome, generate_categories
import numpy as np
import matplotlib.pyplot as plt
import os
from humancompatible.train.optim import SSG
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from humancompatible.train.optim import SSLALM_Adam
import sys, os
from humancompatible.train.optim.PBM import PBM
from fairret.statistic import PositiveRate
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from humancompatible.train.fairness.utils.dataset_loader import get_data_dutch
import torchvision.transforms as transforms
import torchvision
from benchmarking import cifar_train
import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18

def plot_accuracy_per_epoch_std(
    accuracy_mean,
    accuracy_std,
    titles=None,
    eval_points=1,
    train_test='Train',
    save_path="./data/figs/cifar100_benchacc_",
):
    """
    Plots overall accuracy and accuracy per class per epoch (mean ± std).

    Parameters:
    - accuracy_mean: List of length N, each element is a list of K dicts (mean accuracy per class).
    - accuracy_std:  List of length N, each element is a list of K dicts (std accuracy per class).
    - titles: List of titles for each algorithm.
    - eval_points: Evaluate points for markers.
    """

    # --- Color palette (Tableau 10) ---
    colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AB",
    ]

    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    num_algos = len(accuracy_mean)
    if titles is None:
        titles = [f"Algorithm {i + 1}" for i in range(num_algos)]

    fig, axes = plt.subplots(num_algos + 1, 1, figsize=(9, 4 * (num_algos + 1)), sharex=True)

    # ===== Overall accuracy =====
    ax_overall = axes[0]
    for i, (algo_mean, algo_std) in enumerate(zip(accuracy_mean, accuracy_std)):
        K = len(algo_mean)
        x = np.arange(1, K + 1)

        overall_mean = np.array([np.mean(list(epoch.values())) for epoch in algo_mean])
        overall_std = np.array([np.mean(list(epoch.values())) for epoch in algo_std])

        color = colors[i % len(colors)]

        ax_overall.plot(x, overall_mean, lw=2.2, color=color, label=titles[i])
        ax_overall.fill_between(
            x,
            overall_mean - overall_std,
            overall_mean + overall_std,
            color=color,
            alpha=0.15,
        )

        if eval_points is not None:
            idx = (
                np.arange(0, len(overall_mean), eval_points)
                if isinstance(eval_points, int)
                else np.array(eval_points)
            )
            idx = idx[idx < len(overall_mean)]
            ax_overall.plot(
                x[idx],
                overall_mean[idx],
                marker_styles[i % len(marker_styles)],
                color=color,
                markersize=6,
                alpha=0.8,
            )

    ax_overall.set_title(f"{train_test} Accuracy per Algorithm")
    ax_overall.set_ylabel(f"{train_test} Accuracy")
    ax_overall.grid(True, linestyle="--", alpha=0.35)
    ax_overall.legend(fontsize=9, loc="upper left")

    # ===== Per-class accuracy =====
    for i, (ax, algo_mean, algo_std) in enumerate(
        zip(axes[1:], accuracy_mean, accuracy_std)
    ):
        K = len(algo_mean)
        labels = list(algo_mean[0].keys())
        x = np.arange(1, K + 1)

        for j, label in enumerate(labels):
            color = colors[j % len(colors)]

            y_mean = np.array([epoch[label] for epoch in algo_mean])
            y_std = np.array([epoch[label] for epoch in algo_std])

            ax.plot(x, y_mean, lw=2.2, color=color, label=label)
            ax.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                color=color,
                alpha=0.2,
            )

            if eval_points is not None:
                idx = (
                    np.arange(0, len(y_mean), eval_points)
                    if isinstance(eval_points, int)
                    else np.array(eval_points)
                )
                idx = idx[idx < len(y_mean)]
                ax.plot(
                    x[idx],
                    y_mean[idx],
                    marker_styles[j % len(marker_styles)],
                    color=color,
                    markersize=6,
                    alpha=0.8,
                )

        ax.set_title(titles[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy Per Class")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    # plt.savefig(save_path + train_test + ".pdf")
    plt.show()


def plot_accuracy_per_epoch(algorithms_data, titles=None, eval_points=1, train_test='Train',
    save_path="./data/figs/cifar10_benchacc_"):
    """
    Plots overall accuracy and accuracy per class per epoch for each algorithm.

    Parameters:
    - algorithms_data: List of length N, each element is a list of K dictionaries (one per epoch).
    - titles: List of titles for each algorithm (default: "Algorithm 1", ...).
    - eval_points: Evaluate points for markers (default: 1).
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

    num_algos = len(algorithms_data)
    if titles is None:
        titles = [f"Algorithm {i + 1}" for i in range(num_algos)]

    # --- Layout: Add one more subplot for overall accuracy ---
    fig, axes = plt.subplots(num_algos + 1, 1, figsize=(9, 4*(num_algos + 1)), sharex=True)
    # --- Plot overall accuracy ---
    ax_overall = axes[0]
    for i, algorithm_epochs in enumerate(algorithms_data):
        K = len(algorithms_data[0])
        labels = list(algorithm_epochs[0].keys())
        overall_acc = np.array([np.mean(list(epoch.values())) for epoch in algorithm_epochs])
        x = np.arange(1, K+1)
        ax_overall.plot(x, overall_acc, lw=2.2, color=colors[i % len(colors)], label=titles[i])

        if eval_points is not None:
            idx = (
                np.arange(0, len(overall_acc), eval_points)
                if isinstance(eval_points, int)
                else np.array(eval_points)
            )
            idx = idx[idx < len(overall_acc)]
            ax_overall.plot(
                x[idx],
                overall_acc[idx],
                marker_styles[i % len(marker_styles)],
                color=colors[i % len(colors)],
                markersize=6,
                alpha=0.8,
            )

    ax_overall.set_title(f"{train_test} Accuracy per Algorithm")
    ax_overall.set_xlabel("Epoch")
    ax_overall.set_ylabel(f"{train_test} Accuracy")
    ax_overall.grid(True, linestyle="--", alpha=0.35)
    ax_overall.legend(fontsize=9, loc='upper left')

    # --- Plot per-class accuracy ---
    for i, (ax, algorithm_epochs) in enumerate(zip(axes[1:], algorithms_data)):
        K = len(algorithm_epochs)
        labels = list(algorithm_epochs[0].keys())

        for j, label in enumerate(labels):
            color = colors[j % len(colors)]
            y = np.array([epoch[label] for epoch in algorithm_epochs])
            x = np.arange(1, K+1)
            ax.plot(x, y, lw=2.2, color=color, label=label)

            if eval_points is not None:
                idx = (
                    np.arange(0, len(y), eval_points)
                    if isinstance(eval_points, int)
                    else np.array(eval_points)
                )
                idx = idx[idx < len(y)]
                ax.plot(
                    x[idx],
                    y[idx],
                    marker_styles[j % len(marker_styles)],
                    color=color,
                    markersize=6,
                    alpha=0.8,
                )

        ax.set_title(titles[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy Per Class")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()

    # plt.savefig(save_path + train_test + ".pdf")
    plt.show()



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
    save_path="./data/figs/cifar100.pdf",
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
                for i, c_mean in enumerate(constraints):

                    label = titles[j] if i == 0 else None
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
    # plt.show()

def dict_to_array_classes_pergroup(dict_arr, classes):

    n_epochs = len(dict_arr)
    arr = np.zeros((n_epochs, len(classes)))
    for epoch_i, dict_classes in enumerate(dict_arr):
        for class_idx, class_name in enumerate(classes):
            
            arr[epoch_i, class_idx] = dict_arr[epoch_i][class_name]

    return arr

def array_to_dict_classes_pergroup(arr, classes):

    dict_ret = [ ]

    for epoch_i in range(0, len(arr)):
        
        dict_ret.append( {} )
        for class_idx, class_name in enumerate(classes):
            print(class_name)
            print(epoch_i)
            dict_ret[epoch_i][class_name] = arr[epoch_i][class_idx]

    return dict_ret

def array_to_dict_classes_pergroup_all(arr, classes):

    dict_ret = [ ]

    for alg_i in range(0, len(arr)):
        dict_ret.append([])
        for epoch_i in range(0, len(arr[alg_i])):
            dict_ret[alg_i].append( {} )
            for class_idx, class_name in enumerate(classes):
                dict_ret[alg_i][epoch_i][class_name] = arr[alg_i][epoch_i][class_idx]

    return dict_ret

def benchmark(n_epochs, n_constraints, seeds, savepath, dataloader_train, dataloader_test, threshold, classes, class_ind, mu, method_f):

    losses_log = np.zeros((len(seeds), n_epochs))
    constraints_log = np.zeros((len(seeds), n_epochs, n_constraints))
    losses_log_t = np.zeros((len(seeds), n_epochs))
    constraints_log_t = np.zeros((len(seeds), n_epochs, n_constraints))

    accuracy_log = np.zeros((len(seeds), n_epochs))
    accuracy_log_per_group = np.zeros((len(seeds), n_epochs, len(classes)))
    accuracy_log_t = np.zeros((len(seeds), n_epochs))
    accuracy_log_per_group_t = np.zeros((len(seeds), n_epochs, len(classes)))
    
    times_cur = []
    for idx, seed in enumerate(seeds):

        # time the method
        start = time.time()

        losses_cur, constraints_cur, losses_cur_t, constraints_cur_t, accuracy_plotting, accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t\
                                    = method_f(seed, n_epochs, dataloader_train, dataloader_test, threshold, mu)

        # save the timing per epoch
        end = time.time()
        times_cur.append([(end-start)/(n_epochs-1)])

        losses_log[idx] = losses_cur
        constraints_log[idx] = constraints_cur

        losses_log_t[idx] = losses_cur_t
        constraints_log_t[idx] = constraints_cur_t

        accuracy_log[idx] = accuracy_plotting
        accuracy_log_t[idx] = accuracy_plotting_t
        accuracy_log_per_group[idx] = dict_to_array_classes_pergroup(accuracy_per_class_plotting, classes)
        accuracy_log_per_group_t[idx] = dict_to_array_classes_pergroup(accuracy_per_class_plotting_t, classes)

    print('Time elapsed: ', np.array(times_cur).mean())

    losses = list(np.load(savepath)["losses"])
    constraints = list(np.load(savepath)["constraints"])
    losses_std = list(np.load(savepath)["losses_std"])
    constraints_std = list(np.load(savepath)["constraints_std"])
    losses_t = list(np.load(savepath)["losses_t"])
    constraints_t = list(np.load(savepath)["constraints_t"])
    losses_std_t = list(np.load(savepath)["losses_std_t"])
    constraints_std_t = list(np.load(savepath)["constraints_std_t"])
    times = list(np.load(savepath)['times'])
    accuracy = list(np.load(savepath)["accuracy"])
    accuracy_per_group = list(np.load(savepath)["accuracy_per_group"])
    accuracy_t = list(np.load(savepath)["accuracy_t"])
    accuracy_per_group_t = list(np.load(savepath)["accuracy_per_group_t"])
    accuracy_std = list(np.load(savepath)["accuracy_std"])
    accuracy_per_group_std = list(np.load(savepath)["accuracy_per_group_std"])
    accuracy_t_std = list(np.load(savepath)["accuracy_t_std"])
    accuracy_per_group_t_std = list(np.load(savepath)["accuracy_per_group_t_std"])

    # append
    losses += [losses_log.mean(axis=0)]
    constraints += [constraints_log.mean(axis=0).T]
    losses_std += [losses_log.std(axis=0)]
    constraints_std += [constraints_log.std(axis=0).T]

    losses_t += [losses_log_t.mean(axis=0)]
    constraints_t += [constraints_log_t.mean(axis=0).T]
    losses_std_t += [losses_log_t.std(axis=0)]
    constraints_std_t += [constraints_log_t.std(axis=0).T]
    times += [np.array(times_cur).mean()]

    accuracy += [accuracy_log.mean(axis=0)]
    accuracy_per_group += [accuracy_log_per_group.mean(axis=0)]
    accuracy_t += [accuracy_log_t.mean(axis=0)]
    accuracy_per_group_t += [accuracy_log_per_group.mean(axis=0)]

    accuracy_std += [accuracy_log.std(axis=0)]
    accuracy_per_group_std += [accuracy_log_per_group.std(axis=0)]
    accuracy_t_std += [accuracy_log_t.std(axis=0)]
    accuracy_per_group_t_std += [accuracy_log_per_group.std(axis=0)]
    
    np.savez(
        log_path,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
        losses_t=losses_t,
        constraints_t=constraints_t,
        losses_std_t=losses_std_t,
        constraints_std_t=constraints_std_t,
        accuracy = accuracy,
        accuracy_per_group = accuracy_per_group,
        accuracy_t = accuracy_t,
        accuracy_per_group_t = accuracy_per_group_t,
        accuracy_std = accuracy_std,
        accuracy_per_group_std = accuracy_per_group_std,
        accuracy_t_std = accuracy_t_std,
        accuracy_per_group_t_std = accuracy_per_group_t_std,
        times=times
    )   


def load_data(balanced=False):

    # load the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 400

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)

    # Get class names for CIFAR-100
    global classes
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    class_ind = {}

    # index the classes
    for i, classn in enumerate(classes):
        class_ind[classn] = i

    print(classes)
    print(class_ind)

    # load all data and create a balanced sampler
    X = torch.stack([item[0] for item in trainset])
    targets = torch.tensor([item[1] for item in trainset])

    # create onehot vectors
    groups_onehot = torch.eye(100)[targets]

    # create a train dataset
    dataset_train = torch.utils.data.TensorDataset(X, groups_onehot, targets)

    # create the balanced dataloader
    sampler = BalancedBatchSampler(
        group_onehot=groups_onehot, batch_size=batch_size, drop_last=True
    )
    if balanced:
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler, num_workers=10)
    else: 
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=10)

    # load all data and create a balanced sampler
    X_test = torch.stack([item[0] for item in testset])
    targets_test = torch.tensor([item[1] for item in testset])

    # create onehot vectors
    groups_onehot_test = torch.eye(100)[targets_test]

    # split test / val
    X_test, X_val, targets_test, targets_val, groups_onehot_test, groups_onehot_val = \
                            train_test_split(X_test, targets_test, groups_onehot_test, test_size=0.5)

    # create a train dataset
    dataset_test = torch.utils.data.TensorDataset(X_test, groups_onehot_test, targets_test)

    # create the balanced dataloader
    sampler = BalancedBatchSampler(
        group_onehot=groups_onehot_test, batch_size=batch_size, drop_last=True
    )
    global testloader   
    if balanced:
        testloader = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler, num_workers=10)
    else:
        testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, num_workers=10)

    # clean the memory of redundant variables
    del X, targets, groups_onehot
    del X_test, targets_test, groups_onehot_test

    return trainloader, testloader, classes, class_ind

def loss_per_class_f(batch_outputs, batch_targets, network, criterion, num_classes=100):
    """
    Computes the constraint of a demographic parity - that is a loss between all groups
    """

    losses_per_class = torch.zeros(num_classes)

    # for each class compute a loss
    for class_number in range(0, num_classes):

        # get data of that class
        class_args_i = torch.where(batch_targets == class_number)
        batch_outputs_class_i = batch_outputs[class_args_i]
        batch_targets_class_i = batch_targets[class_args_i]
        
        # compute loss for a given class
        batch_loss_class_i = criterion(batch_outputs_class_i, batch_targets_class_i)

        # save the loss
        losses_per_class[class_number] = batch_loss_class_i

    return losses_per_class

def test_network(network):
    
    # prepare count for correct answers
    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    loss_log = []
    c_log = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():

        for data in testloader:
            
            images, _, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # calculate outputs by running images through the network
            outputs = network(images)

            # compute loss per class
            loss_per_class = loss_per_class_f(outputs, labels, network, criterion)

            # compute the loss
            loss = criterion(outputs, labels)   

            # save the logs
            loss_log.append(loss.detach().cpu().numpy())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            
            # compute the accuracy overall
            correct += (predicted == labels).sum().item()

            # compute per class accuracy for this batch
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1

                total_pred[classes[label]] += 1

            # compute the demographic parity constraints
            c_log.append([])
            constraint_k = 0
            for group_i in range(0, 100):
                for group_j in range(0, 100):

                    if group_i != group_j:
                        
                        # demographic parity between i,j
                        g = loss_per_class[group_i] - loss_per_class[group_j]

                        c_log[-1].append(g.detach().cpu().numpy())
                        constraint_k += 1

    # compute the accuracy per class
    for classname, correct_count in correct_pred.items():
        accuracy = float(correct_count) / total_pred[classname]
        correct_pred[classname] = accuracy

    accuracy_total = correct / total
    accuracy_per_group = correct_pred 

    # returns loss, constraints, total hard accuracy, hard accuracy per group, soft accuracy per group
    return loss_log, c_log, accuracy_total, accuracy_per_group

def print_groups(tensor_per_group_acc, classes, class_ind):
    """Accuracy with class labels
    """

    correct_pred = {classname: 0 for classname in classes}
    for classname, _ in correct_pred.items():
            correct_pred[ classname ] = tensor_per_group_acc[class_ind[classname]].detach().cpu().item()
    
    return correct_pred


import torch.nn as nn
import torch.nn.functional as F

# define the network
class Net(nn.Module):
    def __init__(self, _=None, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def adam(seed_n, n_epochs, trainloader, dataloader_test, fair_crit_bound, _):

    # define the network architecture for all classes here
    network_arch = resnet18

    # define the criterion
    global criterion
    criterion = nn.CrossEntropyLoss()
    

    # define the length of the print
    print_n = len(trainloader)

    # define the params and the number of epochs 
    lrs =[0.001]

    # best 
    best_params = None

    # we found the best parameters before this 
    for lr in lrs:
            
            # set the model params
            best_params = {'lr': lr}

    # train the model on cifar dataset, with the best fit
    S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
    test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
    accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
            cifar_train(network_arch, n_epochs, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='unconstrained', model_params=best_params)

    return S_loss_log_plotting, S_c_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, accuracy_plotting, accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t


def ssw(seed_n, n_epochs, trainloader, dataloader_test, fair_crit_bound, _):

    network_arch = resnet18

    # define the criterion
    global criterion
    criterion = nn.CrossEntropyLoss()
    
    # define the length of the print
    print_n = len(trainloader)

    # define the params and the number of epochs 
    lrs =[0.0008]    
    dual_lrs = [0.0008]

    # best 
    best_params = None

    for lr in lrs:
        for dual_lr in dual_lrs:
            
            # set the model params
            best_params = {'lr': lr, 'dual_lr': dual_lr}


    # train the model on cifar dataset, with constraints based on the given parameters and the method
    S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
    test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
    accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
            cifar_train(network_arch, n_epochs, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='ssw', model_params=best_params)

    return S_loss_log_plotting, S_c_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, accuracy_plotting, accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t

def sslalm(seed_n, n_epochs, trainloader, dataloader_test, fair_crit_bound, _):

    network_arch = resnet18

    # define the criterion
    global criterion
    criterion = nn.CrossEntropyLoss()
    
    # define the length of the print
    print_n = len(trainloader)

    lrs = [0.0013]
    dual_lrs = [0.0008]
    mus = [0.1]

    for lr in lrs:
        for dual_lr in dual_lrs:
            for mu in mus:
            
                    # set the model params
                    best_params = {'lr': lr, 'dual_lr': dual_lr, 'mu': mu}

    # train the model on cifar dataset, with constraints based on the given parameters and the method
    S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
    test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
    accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
            cifar_train(network_arch, n_epochs, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='ssl-alm', model_params=best_params)

    return S_loss_log_plotting, S_c_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, accuracy_plotting, accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t


def pbm(seed_n, n_epochs, trainloader, dataloader_test, fair_crit_bound, mu):

    network_arch = resnet18

    # define the criterion
    global criterion
    criterion = nn.CrossEntropyLoss()
    
    # define the length of the print
    print_n = len(trainloader)

    lrs =[0.0018]
    dual_betas = [0.9]
    mus = [mu]
    init_duals = [0.001]
    # penalties = ["quadratic_logarithmic", "quadratic_reciprocal"]
    penalties = ["quadratic_logarithmic"]
    warm_starts = [2]

    for lr in lrs:
        for dual_beta in dual_betas:
            for penalty in penalties:
                    for init_dual in init_duals:
                            for mu in mus:
                                    for warm_start in warm_starts:
                                            # set the model params
                                            best_params  = {'lr': lr, 'dual_beta': dual_beta, 'mu': mu, 'penalty': penalty, 'init_dual': init_dual, 'warm_start': warm_start}

    # train the model on cifar dataset, with constraints based on the given parameters and the method
    S_loss_log_plotting, S_c_log_plotting, S_loss_std_log_plotting, S_c_std_log_plotting,\
    test_S_loss_log_plotting, test_S_c_log_plotting, test_S_loss_std_log_plotting, test_S_c_std_log_plotting,\
    accuracy_plotting,  accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t = \
            cifar_train(network_arch, n_epochs, seed_n, trainloader, loss_per_class_f, test_network, device, classes, fair_crit_bound, print_n, method='pbm', model_params=best_params)

    return S_loss_log_plotting, S_c_log_plotting, test_S_loss_log_plotting, test_S_c_log_plotting, accuracy_plotting, accuracy_per_class_plotting, accuracy_plotting_t, accuracy_per_class_plotting_t

if __name__ == '__main__':

    # define the torch seed here
    n_epochs = 30
    n_constraints = 9900
    threshold = 0.5
    # device = 'cpu'    
    device = 'cuda:0'
    bench_mus = False  # true to benchmark mus on cifar10 pbm
    print(torch.version.cuda)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # define seeds
    seeds = [1, 2, 3]


    # log path file
    if bench_mus:
        log_path = "./data/logs/cifar100_bench_mus.npz"
    else: 
        log_path = "./data/logs/cifar100_bench.npz"


    # load data
    trainloader, testloader, classes, class_ind = load_data(balanced=True)

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
        times=[],
        accuracy=[],
        accuracy_per_group=[],
        accuracy_t=[],
        accuracy_per_group_t=[],
        accuracy_std=[],
        accuracy_per_group_std=[],
        accuracy_t_std=[],
        accuracy_per_group_t_std=[]
    )

    if not bench_mus:

        print('Starting cifar100 benchmark')

        # benchmark adam
        benchmark(n_epochs, n_constraints, seeds, log_path, trainloader, testloader, threshold, classes, class_ind, 0, adam)
        print('ADAM DONE!!!')

        # benchmark ssw
        benchmark(n_epochs, n_constraints, seeds, log_path, trainloader, testloader, threshold, classes, class_ind, 0, ssw)
        print('SSW DONE!!!')

        # # benchmark sslalm
        benchmark(n_epochs, n_constraints, seeds, log_path, trainloader, testloader, threshold, classes, class_ind, 0, sslalm)
        print('SSLALM DONE!!!')

        # #  benchmark pbm
        mu = 1.0
        benchmark(n_epochs, n_constraints, seeds, log_path, trainloader, testloader, threshold, classes, class_ind, mu, pbm)
        print('PBM DONE!!!')

    else: 

        print('Starting cifar100 mus benchmark')

        titles = []
        seeds = [1, 2, 3]
        mus = [0.0, 0.1, 0.5, 1.0]

        for mu in mus: 
            benchmark(n_epochs, n_constraints, seeds, log_path, trainloader, testloader, threshold, classes, class_ind, mu, pbm)
            print(f'PBM {mu} DONE!!!')
            titles.append( f"SPBM_mu={mu}")


    # PLOT 
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])
    losses_t = list(np.load(log_path)["losses_t"])
    constraints_t = list(np.load(log_path)["constraints_t"])
    losses_std_t = list(np.load(log_path)["losses_std_t"])
    constraints_std_t = list(np.load(log_path)["constraints_std_t"])
    times = list(np.load(log_path)['times'])
    accuracy = list(np.load(log_path)["accuracy"])
    accuracy_per_group = list(np.load(log_path)["accuracy_per_group"])
    accuracy_t = list(np.load(log_path)["accuracy_t"])
    accuracy_per_group_t = list(np.load(log_path)["accuracy_per_group_t"])
    accuracy_std = list(np.load(log_path)["accuracy_std"])
    accuracy_per_group_std = list(np.load(log_path)["accuracy_per_group_std"])
    accuracy_t_std = list(np.load(log_path)["accuracy_t_std"])
    accuracy_per_group_t_std = list(np.load(log_path)["accuracy_per_group_t_std"])


    if not bench_mus:
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
                "SSL-ALM",
                "SPBM"
            ],
            log_constraints=False,
            std_multiplier=1,
            mode='train_test', # change this to 'train', to ignore the test=
            plot_time_instead_epochs=False,
            save_path="./data/figs/cifar100_bench.pdf"
        )
        
        accuracy_per_group = array_to_dict_classes_pergroup_all(accuracy_per_group, classes)
        accuracy_per_group_std = array_to_dict_classes_pergroup_all(accuracy_per_group_std, classes)
        # plot_accuracy_per_epoch(accuracy_per_group, titles=["Unconstrained Adam", "SSW", "SSL-ALM", f"SPBM"], train_test='Train')
        plot_accuracy_per_epoch_std(accuracy_per_group, accuracy_per_group_std, titles=["Unconstrained Adam", "SSW", "SSL-ALM", f"SPBM"], train_test='Train', save_path="./data/figs/cifar100_benchacc_")

        accuracy_per_group_t = array_to_dict_classes_pergroup_all(accuracy_per_group_t, classes)
        accuracy_per_group_t_std = array_to_dict_classes_pergroup_all(accuracy_per_group_t_std, classes)
        plot_accuracy_per_epoch_std(accuracy_per_group_t, accuracy_per_group_t_std, titles=["Unconstrained Adam", "SSW", "SSL-ALM", f"SPBM"], train_test='Test', save_path="./data/figs/cifar100_benchacc_")


    else: 
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
            titles=titles,
            log_constraints=False,
            std_multiplier=1,
            mode='train_test', # change this to 'train', to ignore the test=
            plot_time_instead_epochs=False,
            save_path="./data/figs/cifar100_mus_bench.pdf"
        )

        accuracy_per_group = array_to_dict_classes_pergroup_all(accuracy_per_group, classes)
        accuracy_per_group_std = array_to_dict_classes_pergroup_all(accuracy_per_group_std, classes)
        plot_accuracy_per_epoch(accuracy_per_group, titles=titles, train_test='Train')
        plot_accuracy_per_epoch_std(accuracy_per_group, accuracy_per_group_std, titles=titles, train_test='Train', save_path="./data/figs/cifar100_mus_benchacc_")

        accuracy_per_group_t = array_to_dict_classes_pergroup_all(accuracy_per_group_t, classes)
        accuracy_per_group_t_std = array_to_dict_classes_pergroup_all(accuracy_per_group_t_std, classes)
        plot_accuracy_per_epoch_std(accuracy_per_group_t, accuracy_per_group_t_std, titles=titles, train_test='Test', save_path="./data/figs/cifar100_mus_benchacc_")