"""
python script with eq. opportunity benchmarking
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
    save_path=None
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

            c_min = np.min(constraints - std_multiplier * constraints_std, axis=0)
            c_max = np.max(constraints + std_multiplier * constraints_std, axis=0)

            ax.fill_between(x, c_min, c_max, color=color, alpha=0.1)

            for c_idx, c_mean in enumerate(constraints):

                if plot_time_instead_epochs:
                    x *= round(times[j])

                if c_idx == 0:
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
    if save_path:
        plt.savefig(save_path)

    plt.show()

def positive_rate(out_batch, batch_sens, prob_f=torch.nn.functional.sigmoid):
    """
    Calculates the positive rate vector based on the given outputs of the model for the given groups. 
    
    """

    # compute the probabilities - using sigmoid (since that is used )
    if prob_f is None: 
        preds = out_batch
    else: 
        preds = prob_f( out_batch )

    pr = PositiveRate()
    probs_per_group = pr(preds, batch_sens)  # P(y=1|Sk = 1)

    return probs_per_group

def load_data():

    # load folktables data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["VA"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSIncome.features, definition_df=definition_df
    )
    df_feat, df_labels, _ = ACSIncome.df_to_pandas(
        acs_data, categories=categories, dummies=True
    )

    # split the data on groups, labels and features - the features should not have the sensitive feature
    sens_cols = [
        "SEX_Female",
        "SEX_Male",
        "MAR_Divorced",
        "MAR_Married",
        "MAR_Never married or under 15 years old",
    ]
    features = df_feat.drop(columns=sens_cols).to_numpy(dtype="float")
    groups = df_feat[sens_cols].to_numpy(dtype="float")
    labels = df_labels.to_numpy(dtype="float")

    # Split columns into sex and marital
    sex_cols = ["SEX_Female", "SEX_Male"]
    mar_cols = [
        "MAR_Divorced",
        "MAR_Married",
        "MAR_Never married or under 15 years old",
    ]

    # Convert each row to sex index and marital index
    sex_idx = df_feat[sex_cols].values.argmax(axis=1)
    mar_idx = df_feat[mar_cols].values.argmax(axis=1)

    # Number of unique combinations
    num_groups = len(sex_cols) * len(mar_cols)

    # Map each combination to a unique index
    group_indices = sex_idx * len(mar_cols) + mar_idx  # shape: (num_samples,)

    # One-hot encode the combinations
    groups_onehot = np.eye(num_groups)[group_indices]

    # Create dictionary mapping index to combination
    group_dict = {}
    for i, (s, m) in enumerate(product(sex_cols, mar_cols)):
        group_dict[i] = f"{s} + {m}"

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups_onehot, test_size=0.2, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # print the statistics
    for idx in group_dict:
        print(f"{group_dict[idx]}, : {(groups_onehot[:, idx] == 1).sum()}")

    # make into a pytorch dataset, remove the sensitive attribute
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train)
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test)
    dataset_test = torch.utils.data.TensorDataset(features_test, labels_test)

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=120, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=120, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    dataloader_train, dataloader_test, features_train

def benchmark(n_epochs, n_constraints, seeds, savepath, dataloader, features_train, threshold, method_f):

    losses_log = np.zeros((len(seeds), n_epochs))
    constraints_log = np.zeros((len(seeds), n_epochs, n_constraints))
    for idx, seed in enumerate(seeds):
        losses_cur, constraints_cur = method_f(seed, n_epochs, dataloader, features_train, threshold)
        losses_log[idx] = losses_cur
        constraints_log[idx] = constraints_cur

    losses = list(np.load(savepath)["losses"])
    constraints = list(np.load(savepath)["constraints"])
    losses_std = list(np.load(savepath)["losses_std"])
    constraints_std = list(np.load(savepath)["constraints_std"])

    # append
    losses += [losses_log.mean(axis=0)]
    constraints += [constraints_log.mean(axis=0).T]
    losses_std += [losses_log.std(axis=0)]
    constraints_std += [constraints_log.std(axis=0).T]

    # save the computed data
    np.savez(savepath, losses=losses, constraints=constraints, losses_std=losses_std, constraints_std=constraints_std)


if __name__ == '__main__':

    # define the torch seed here
    n_epochs = 2
    n_constraints = 1
    threshold = 0.1

    # define seeds
    seeds = [1, 2, 3]

    # log path file
    log_path = "./data/logs/log_benchmark_stochastic_2_bench.npz"

    # load data
    dataloader_train, dataloader_test, features_train= load_data()

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
        constraints_std_t=[]
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
    save_path="./data/figs/ACSIncome_equal_opportunity_bench.pdf"
    )