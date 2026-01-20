
"""
python script with acsincome vector benchmarking
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
import time

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
    mode='train_test',  # "train" or "train_test"
    times=[], # second per epoch
    plot_time_instead_epochs=False,
    save_path="./data/figs/ACSIncome_vector_bench.pdf"
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
    # plt.show()


def test_model(model, test_dataloader, fair_criterion, criterion):

    with torch.no_grad():

        t_loss_log_plotting = []  # mean
        t_c_log_plotting = []  # mean

        # go though all data
        for batch_input, batch_sens, batch_label in test_dataloader:

            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # save the fair loss violation for logging
            t_c_log_plotting.append([fair_loss.detach().item()])

            # calculate primal loss and grad
            loss = criterion(out, batch_label)
            
            t_loss_log_plotting.append(loss.detach().numpy())

    return t_loss_log_plotting, t_c_log_plotting


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

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=64, drop_last=True
    )
    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test)
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=64, drop_last=True
    )

    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    return dataloader_train, dataloader_test, features_train

def adam(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, fair_crit_bound):

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

    # create the SSLALM optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)

    # alloc arrays for plotting
    Adam_S_loss_log_plotting = []  # mean
    Adam_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)
    loss_log, c_log = test_model(model, dataloader_train, fair_criterion, criterion)

    Adam_S_c_log_plotting.append(np.mean(c_log, axis=0))
    Adam_S_loss_log_plotting.append(np.mean(loss_log))

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    # training loop
    for epoch in range(n_epochs-1):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader_train:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # calculate the fair constraint violation
            fair_constraint = fair_loss

            # save the fair loss violation for logging
            c_log.append([fair_loss.detach().item()])

            # calculate primal loss and grad
            loss = criterion(out, batch_label)
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)

        Adam_S_c_log_plotting.append(np.mean(c_log, axis=0))
        Adam_S_loss_log_plotting.append(np.mean(loss_log))

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints Train/Test: {np.mean(c_log, axis=0)}/{np.mean(t_c, axis=0)}, "
        )
    
    return Adam_S_loss_log_plotting, Adam_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting

def ssw(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, fair_crit_bound):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # same network size for all algorithms
    hsize1 = 64
    hsize2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    optimizer = SSG(params=model.parameters(), m=1, lr=0.1, dual_lr=0.05)

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)
    loss_log, c_log = test_model(model, dataloader_train, fair_criterion, criterion)

    SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
    SSG_S_loss_log_plotting.append(np.mean(loss_log))

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    # training loop
    for epoch in range(n_epochs-1):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader_train:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)
            fair_constraint = torch.max(fair_loss - fair_crit_bound, torch.zeros(1))
            fair_constraint.backward(retain_graph=True)

            # compute the grad of the constraints
            optimizer.dual_step(0)
            optimizer.zero_grad()

            # save the constraint value
            c_log.append([fair_loss.detach().item()])

            # calculate loss and grad
            loss = criterion(out, batch_label)
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step(fair_constraint)
            optimizer.zero_grad()

        SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSG_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints Train/Test: {np.mean(c_log, axis=0)}/{np.mean(t_c, axis=0)}, "
        )

    return SSG_S_loss_log_plotting, SSG_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting



def sslalm(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, fair_crit_bound):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 64
    hsize2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )
        
    optimizer = SSLALM_Adam(
        params=model.parameters(),
        m=1,  # number of constraints - one in our case
        lr=0.001,  # primal variable lr
        dual_lr=0.05,  # lr of a dual ALM variable
        dual_bound=5,
        rho=1,  # rho penalty in ALM parameter
        mu=2,  # smoothing parameter
    )

    # define epochs + loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)

    # add slack variables - to create the equality from the inequalities    
    slack_vars = torch.zeros(1, requires_grad=True)
    optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})

        # alloc arrays for plotting
    SSLALM_S_loss_log_plotting = []  # mean
    SSLALM_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)
    loss_log, c_log = test_model(model, dataloader_train, fair_criterion, criterion)

    SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
    SSLALM_S_loss_log_plotting.append(np.mean(loss_log))

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    # training loop
    for epoch in range(n_epochs-1):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader_train:
            constr = torch.zeros(1, dtype=torch.float32)
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # calculate the fair constraint violation
            fair_constraint = fair_loss + slack_vars[0] - fair_crit_bound
            # calculate the fair constraint violation
            # fair_constraint = torch.maximum(fair_loss - fair_crit_bound, torch.zeros(1) )[0]

            # perform the dual step variable + save the dual grad for later
            optimizer.dual_step(0, c_val=fair_constraint)
            constr[0] = fair_constraint

            # save the fair loss violation for logging
            c_log.append([fair_loss.detach().item()])
            duals_log.append(optimizer._dual_vars.detach())

            # calculate primal loss and grad
            loss = criterion(out, batch_label) + 0 * slack_vars[0]
            # loss = criterion(out, batch_label) 
            loss_log.append(loss.detach().numpy())
            optimizer.step(loss, constr)
            
            # slack variables must be non-negative. this is the "projection" step from the SSL-ALM paper
            with torch.no_grad():
                for s in slack_vars:
                    if s < 0:
                        s.zero_()

        SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSLALM_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints Train/Test: {np.mean(c_log, axis=0)}/{np.mean(t_c, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )
    
    return SSLALM_S_loss_log_plotting, SSLALM_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting


def pbm(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, fair_crit_bound):

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

    # create the PBM optimizer
    optimizer = PBM(params=model.parameters(), m=1, lr=0.001, dual_beta=0.95, mu=0.1, penalty_update_m='CONST',
                    barrier="quadratic_logarithmic", epoch_len=len(dataloader_train))

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)
    loss_log, c_log = test_model(model, dataloader_train, fair_criterion, criterion)

    SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
    SSG_S_loss_log_plotting.append(np.mean(loss_log))

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    # training loop
    for epoch in range(n_epochs-1):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader_train:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)
            fair_constraint = fair_loss - fair_crit_bound

            # compute the grad of the constraints
            optimizer.dual_step(0, fair_constraint)

            # save the constraint value
            c_log.append([fair_loss.detach().item()])
            duals_log.append(optimizer._dual_vars.detach())

            # calculate loss and grad
            loss = criterion(out, batch_label)
            loss_log.append(loss.detach().numpy())
            optimizer.step(loss)

        SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSG_S_loss_log_plotting.append(np.mean(loss_log))


        # test the model
        t_loss, t_c = test_model(model, dataloader_test, fair_criterion, criterion)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints Train/Test: {np.mean(c_log, axis=0)}/{np.mean(t_c, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )

    return SSG_S_loss_log_plotting, SSG_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting


def benchmark(n_epochs, n_constraints, seeds, savepath, dataloader_train, dataloader_test, features_train, threshold, method_f):

    losses_log = np.zeros((len(seeds), n_epochs))
    constraints_log = np.zeros((len(seeds), n_epochs, n_constraints))
    losses_log_t = np.zeros((len(seeds), n_epochs))
    constraints_log_t = np.zeros((len(seeds), n_epochs, n_constraints))
    times_cur = []
    for idx, seed in enumerate(seeds):

        # time the method
        start = time.time()

        losses_cur, constraints_cur, losses_cur_t, constraints_cur_t = method_f(seed, n_epochs, dataloader_train, dataloader_test, features_train, threshold)

        # save the timing per epoch
        end = time.time()
        times_cur.append([(end-start)/(n_epochs-1)])

        losses_log[idx] = losses_cur
        constraints_log[idx] = constraints_cur

        losses_log_t[idx] = losses_cur_t
        constraints_log_t[idx] = constraints_cur_t

    losses = list(np.load(savepath)["losses"])
    constraints = list(np.load(savepath)["constraints"])
    losses_std = list(np.load(savepath)["losses_std"])
    constraints_std = list(np.load(savepath)["constraints_std"])
    losses_t = list(np.load(savepath)["losses_t"])
    constraints_t = list(np.load(savepath)["constraints_t"])
    losses_std_t = list(np.load(savepath)["losses_std_t"])
    constraints_std_t = list(np.load(savepath)["constraints_std_t"])
    times = list(np.load(savepath)['times'])

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

    # save the computed data
    np.savez(savepath, losses=losses, constraints=constraints, losses_std=losses_std, constraints_std=constraints_std,
             losses_t=losses_t, constraints_t=constraints_t, losses_std_t=losses_std_t, constraints_std_t=constraints_std_t, times=times)


if __name__ == '__main__':

    # define the torch seed here
    n_epochs = 6
    n_constraints = 1
    threshold = 0.1

    # define seeds
    seeds = [1, 2, 3]

    # log path file
    log_path = "./data/logs/log_benchmark_stochastic_bench.npz"

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
        constraints_std_t=[],
        times=[]
    )

    # benchmark adam
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, adam)
    print('ADAM DONE!!!')

    # benchmark ssw
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, ssw)
    print('SSW DONE!!!')

    # benchmark sslalm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, sslalm)
    print('SSLALM DONE!!!')

    # benchmark pbm
    benchmark(n_epochs, n_constraints, seeds, log_path, dataloader_train, dataloader_test, features_train, threshold, pbm)
    print('PBM DONE!!!')

    print('times:', list(np.load(log_path)["times"]))

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
        save_path="./data/figs/ACSIncome_vector.pdf"
    )