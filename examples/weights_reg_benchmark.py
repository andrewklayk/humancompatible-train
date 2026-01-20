

"""
python script with weight reg benchmarking
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
    save_path="./data/figs/weight_reg_bench.pdf"
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


def load_data():

    # load and prepare data

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

    sens_cols = ["SEX_Female", "SEX_Male"]
    features = df_feat.drop(columns=sens_cols).to_numpy(dtype="float")
    groups = df_feat[sens_cols].to_numpy(dtype="float")
    labels = df_labels.to_numpy(dtype="float")

    print(sens_cols)
    print(features.shape)
    print(groups.shape)
    print(labels.shape)

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups, test_size=0.2, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # make into a pytorch dataset, remove the sensitive attribute
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test)

    # create a dataset
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    return dataloader, features_train


def adam(seed_n, n_epochs, dataloader, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # create small FC network
    latent_size1 = 64
    latent_size2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    # get number of layers + number of biases
    m = len(list(model.parameters()))

    # create the SSLALM optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    # bounds for the constraints: norm of each weight matrix should be <= 1
    constraint_bounds = [threshold] * m

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # alloc arrays for plotting
    Adam_loss_log_plotting = []
    Adam_c_log_plotting = []

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays
        loss_log = []
        c_log = []

        # go through all data
        for batch_input, batch_label in dataloader:
            # calculate constraints - just for logging since this is an unconstrainted optimization
            c_log.append([])
            for i, param in enumerate(model.parameters()):
                # norm of the w. matrix
                norm = torch.linalg.norm(param, ord=2)

                # convert constraint to equality
                norm_viol = torch.max(norm - constraint_bounds[i], torch.zeros(1))

                # save the value of the constraint
                c_log[-1].append(norm.detach().numpy())

            # calculate loss and grad
            batch_output = model(batch_input)
            loss = criterion(batch_output, batch_label)
            loss.backward()

            # save the loss and the dual variables
            loss_log.append(loss.detach().numpy())

            # update the primal variables together with smoothing dual variable
            optimizer.step()
            optimizer.zero_grad()

        # save the epoch values for plotting
        Adam_loss_log_plotting.append(np.mean(loss_log))
        Adam_c_log_plotting.append(np.mean(c_log, axis=0))

        # print out the epoch values
        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )

    return Adam_loss_log_plotting, Adam_c_log_plotting

def ssw(seed_n, n_epochs, dataloader, features_train, threshold):

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # create small FC network - same as the other algorithms
    latent_size1 = 64
    latent_size2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    # get number of layers + number of biases
    m = len(list(model.parameters()))

    # create the SSLALM optimizer
    optimizer = SSG(params=model.parameters(), m=1, lr=0.01, dual_lr=0.1)

    # bounds for the constraints: norm of max each weight matrix should be <= 1
    constraint_bounds = [threshold]

    # define epochs + loss function - same loss should be defined for all algorithms
    criterion = torch.nn.BCEWithLogitsLoss()

    # alloc the plotting array
    SSG_c_log_plotting = []
    SSG_loss_log_plotting = []

    # training loop
    for epoch in range(n_epochs):
        # alloc logging array
        loss_log = []
        c_log = []
        duals_log = []

        # train for all data
        for batch_input, batch_label in dataloader:
            # prepare the max of the violation
            max_norm_viol = torch.zeros(1)
            c_log.append([])

            # calculate constraints and constraint grads - max of constraint per each weight matrix
            for i, param in enumerate(model.parameters()):
                # norm of the w. matrix
                norm = torch.linalg.norm(param, ord=2)

                # convert constraint to equality
                norm_viol = torch.max(norm - constraint_bounds[0], torch.zeros(1))

                # save the max
                max_norm_viol = torch.max(max_norm_viol, norm_viol)

                # save the value of the constraint
                c_log[-1].append(norm.detach().numpy())

            # calculate the Jacobian of the max-violating norm constraint
            max_norm_viol.backward()

            # save the gradient of the constraint
            optimizer.dual_step(0)
            optimizer.zero_grad()

            # calculate loss and grad
            out = model(batch_input)
            loss = criterion(out, batch_label)
            loss.backward()

            # save the loss value
            loss_log.append(loss.detach().numpy())

            # perform a step - either update based on the loss grad or constraint grad
            optimizer.step(max_norm_viol)
            optimizer.zero_grad()

        # save the epoch values for plotting
        SSG_loss_log_plotting.append(np.mean(loss_log))
        SSG_c_log_plotting.append(np.mean(c_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )

    return SSG_loss_log_plotting, SSG_c_log_plotting


def sslalm(seed_n, n_epochs, dataloader, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # create small FC network
    latent_size1 = 64
    latent_size2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    # get number of layers + number of biases
    m = len(list(model.parameters()))

    # create the SSLALM optimizer
    optimizer = SSLALM_Adam(params=model.parameters(), m=m, lr=0.001, dual_lr=0.1)

    # bounds for the constraints: norm of each weight matrix should be <= 1
    constraint_bounds = [threshold] * m

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # alloc arrays for plotting
    SSL_ALM_Adam_loss_log_plotting = []
    SSL_ALM_Adam_c_log_plotting = []

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays
        loss_log = []
        c_log = []
        duals_log = []

        # go through all data
        for batch_input, batch_label in dataloader:
            # calculate constraints and constraint grads - constraint per each weight matrix
            c_log.append([])
            for i, param in enumerate(model.parameters()):
                # norm of the w. matrix
                norm = torch.linalg.norm(param, ord=2)

                # convert constraint to equality
                norm_viol = torch.max(norm - constraint_bounds[i], torch.zeros(1))

                # update the dual variable + save the Jacobian for later - its needed in the primal variable update
                optimizer.dual_step(i, c_val=norm_viol[0])

                # save the value of the constraint
                c_log[-1].append(norm.detach().numpy())

            # calculate loss and grad
            batch_output = model(batch_input)
            loss = criterion(batch_output, batch_label)

            # save the loss and the dual variables
            loss_log.append(loss.detach().numpy())
            duals_log.append(optimizer._dual_vars.detach())

            # update the primal variables together with smoothing dual variable
            optimizer.step(loss)

        # save the epoch values for plotting
        SSL_ALM_Adam_loss_log_plotting.append(np.mean(loss_log))
        SSL_ALM_Adam_c_log_plotting.append(np.mean(c_log, axis=0))

        # print out the epoch values
        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )

    return SSL_ALM_Adam_loss_log_plotting, SSL_ALM_Adam_c_log_plotting

def pbm(seed_n, n_epochs, dataloader, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # create small FC network
    latent_size1 = 64
    latent_size2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    # get number of layers + number of biases
    m = len(list(model.parameters()))

    # create the SSLALM optimizer
    optimizer = PBM(params=model.parameters(), m=m, lr=0.001, dual_beta=0.95, penalty_update_m='CONST', barrier="quadratic_logarithmic", epoch_len=len(dataloader))

    # bounds for the constraints: norm of each weight matrix should be <= 1
    constraint_bounds = [threshold] * m

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # alloc arrays for plotting
    PBM_Adam_loss_log_plotting = []
    PBM_Adam_c_log_plotting = []

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays
        loss_log = []
        c_log = []
        duals_log = []

        # go through all data
        for batch_input, batch_label in dataloader:
            # calculate constraints and constraint grads - constraint per each weight matrix
            c_log.append([])
            for i, param in enumerate(model.parameters()):
                # norm of the w. matrix
                norm = torch.linalg.norm(param, ord=2)

                # copmute the constraint
                norm_viol = norm - constraint_bounds[i]
                
                # update the dual variable + save the Jacobian for later - its needed in the primal variable update
                optimizer.dual_step(i, c_val=norm_viol)

                # save the value of the constraint
                c_log[-1].append(norm.detach().numpy())

            # calculate loss and grad
            batch_output = model(batch_input)
            loss = criterion(batch_output, batch_label)

            # save the loss and the dual variables
            loss_log.append(loss.detach().numpy())
            duals_log.append(optimizer._dual_vars.detach())

            # update the primal variables together with smoothing dual variable
            optimizer.step(loss)

        # save the epoch values for plotting
        PBM_Adam_loss_log_plotting.append(np.mean(loss_log))
        PBM_Adam_c_log_plotting.append(np.mean(c_log, axis=0))

        # print out the epoch values
        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )

    return PBM_Adam_loss_log_plotting, PBM_Adam_c_log_plotting

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
    n_constraints = 6
    threshold = 0.2

    # define seeds
    seeds = [1, 2, 3]

    # log path file
    log_path = "./data/logs/weights_reg_bench.npz"

    # load data
    dataloader, features_train = load_data()

    # resave to empty file
    np.savez(
    log_path,
        losses=[],
        constraints=[],
        losses_std=[],
        constraints_std=[]
    )

    # benchmark adam
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader, features_train, threshold, adam)
    print('ADAM DONE!!!')

    # benchmark ssw
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader, features_train, threshold, ssw)
    print('SSW DONE!!!')

    # benchmark sslalm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader, features_train, threshold, sslalm)
    print('SSLALM DONE!!!')

    # benchmark pbm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader, features_train, threshold, pbm)
    print('PBM DONE!!!')

    # PLOT 
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    plot_losses_and_constraints_stochastic(
    losses,
    losses_std,
    constraints,
    constraints_std,
    [threshold],
    titles=["Unconstrained Adam", "SSW", "SSL-ALM", "SPBM"]
    )