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
        group_onehot=sens_train, batch_size=30, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=30, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    return dataloader_train, dataloader_test, features_train

def test_model(model, test_dataloader, criterion, threshold):

    with torch.no_grad():

        t_loss_log_plotting = []  # mean
        t_c_log_plotting = []  # mean

        # go though all data
        for batch_input, batch_sens, batch_label in test_dataloader:

            # calculate constraints and constraint grads
            out = model(batch_input)

            # compute per group positive rate
            pos_rate_pergroup = positive_rate(out, batch_sens, prob_f=torch.nn.functional.sigmoid)

            # prepare counter + array of constr for this batch
            current_constr = 0
            t_c_log_plotting.append([])

            # compute the equal opportunity constraint
            for i in range(0, len(pos_rate_pergroup)):
                for j in range(0, len(pos_rate_pergroup)):

                    # calculate the constraint only for different subgroups
                    if i != j:

                        # the constraint with the slack variables
                        constr_ij = pos_rate_pergroup[i] - pos_rate_pergroup[j] 
                        constr_ij = constr_ij - threshold
                        
                        # save the value of the constraint
                        t_c_log_plotting[-1].append(constr_ij.detach().numpy() + threshold)

                        # iterate the constraint counter
                        current_constr += 1

            # compute the augemented objective loss
            loss = criterion(out, batch_label)

            # save the logs
            t_loss_log_plotting.append(loss.detach().numpy())


    return t_loss_log_plotting, t_c_log_plotting

def adam(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, threshold):
    
    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 128
    hsize2 = 64
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    # alloc arrays for plotting
    adam_S_loss_log_plotting = []  # mean
    adam_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)
    loss_log, c_log = test_model(model, dataloader_train, criterion, threshold)

    adam_S_c_log_plotting.append(np.mean(c_log, axis=0))
    adam_S_loss_log_plotting.append(np.mean(loss_log))

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

            # compute per group positive rate
            pos_rate_pergroup = positive_rate(out, batch_sens, prob_f=torch.nn.functional.sigmoid)

            # prepare counter + array of constr for this batch
            current_constr = 0
            c_log.append([])

            # compute the equal opportunity constraint
            for i in range(0, len(pos_rate_pergroup)):
                for j in range(0, len(pos_rate_pergroup)):

                    # calculate the constraint only for different subgroups
                    if i != j:

                        # the constraint with the slack variables
                        constr_ij = pos_rate_pergroup[i] - pos_rate_pergroup[j] 
                        constr_ij = constr_ij - threshold
                        
                        # save the value of the constraint
                        c_log[-1].append(constr_ij.detach().numpy() + threshold)

                        # iterate the constraint counter
                        current_constr += 1

            # compute the augemented objective loss
            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save the logs
            loss_log.append(loss.detach().numpy())


        adam_S_c_log_plotting.append(np.mean(c_log, axis=0))
        adam_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints: {np.max(np.abs(np.mean(c_log, axis=0)))}/{np.max(np.abs(np.mean(t_c_log_plotting, axis=0)))}, "
        )

    return adam_S_loss_log_plotting, adam_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting


def ssw(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # same network size for all algorithms
    hsize1 = 128
    hsize2 = 32
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    number_of_constraints = 30
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = SSG(params=model.parameters(), m=1, lr=0.01, dual_lr=0.05)

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)
    loss_log, c_log = test_model(model, dataloader_train, criterion, threshold)

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
    SSG_S_loss_log_plotting.append(np.mean(loss_log))

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

            # compute per group positive rate
            pos_rate_pergroup = positive_rate(out, batch_sens, prob_f=torch.nn.functional.sigmoid)

            # prepare counter + array of constr for this batch
            current_constr = 0
            c_log.append([])

            # prepare the max of the constraints
            max_norm_viol = torch.zeros(1)

            # compute the equal opportunity constraint
            for i in range(0, len(pos_rate_pergroup)):
                for j in range(0, len(pos_rate_pergroup)):

                    # calculate the constraint only for different subgroups
                    if i != j:

                        # the constraint with the slack variables
                        constr_ij = pos_rate_pergroup[i] - pos_rate_pergroup[j] 
                        constr_ij = constr_ij - threshold

                        # save the max
                        max_norm_viol = torch.max(max_norm_viol, constr_ij)

                        # save the value of the constraint
                        c_log[-1].append(constr_ij.detach().numpy() + threshold)

                        # iterate the constraint counter
                        current_constr += 1

            # calculate the Jacobian of the max-violating norm constraint
            max_norm_viol.backward(retain_graph=True)

            # save the gradient of the constraint
            optimizer.dual_step(0)
            optimizer.zero_grad()
            
            loss = criterion(out, batch_label)
            loss.backward()
            optimizer.step(max_norm_viol)
            optimizer.zero_grad()

            # save the logs
            loss_log.append(loss.detach().numpy())


        SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSG_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints: {np.max(np.abs(np.mean(c_log, axis=0)))}/{np.max(np.abs(np.mean(t_c_log_plotting, axis=0)))}, "
        )

    return SSG_S_loss_log_plotting, SSG_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting

def sslalm(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 128
    hsize2 = 64
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    # 10*10* 2 constraint - per subgroup inequality + 2 per inequality
    number_of_constraints = 30
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = SSLALM_Adam(
        params=model.parameters(),
        m=number_of_constraints,  # number of constraints - one in our case
        lr=0.001,  # primal variable lr
        dual_lr=0.05,  # lr of a dual ALM variable
        dual_bound=5,
        rho=1,  # rho penalty in ALM parameter
        mu=2,  # smoothing parameter
    )

    # add slack variables - to create the equality from the inequalities    
    slack_vars = torch.zeros(number_of_constraints, requires_grad=True)
    optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})

    # alloc arrays for plotting
    SSLALM_S_loss_log_plotting = []  # mean
    SSLALM_S_c_log_plotting = []  # mean

    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)
    loss_log, c_log = test_model(model, dataloader_train, criterion, threshold)

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
    SSLALM_S_loss_log_plotting.append(np.mean(loss_log))

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

            # compute per group positive rate
            pos_rate_pergroup = positive_rate(out, batch_sens, prob_f=torch.nn.functional.sigmoid)

            # prepare counter + array of constr for this batch
            current_constr = 0
            c_log.append([])

            # compute the equal opportunity constraint
            constraints = torch.zeros(30)
            for i in range(0, len(pos_rate_pergroup)):
                for j in range(0, len(pos_rate_pergroup)):

                    # calculate the constraint only for different subgroups
                    if i != j:

                        # the constraint with the slack variables
                        constr_ij = pos_rate_pergroup[i] - pos_rate_pergroup[j] 
                        constr_ij = constr_ij + slack_vars[current_constr] - threshold

                        # perform the dual step variable + save the dual grad for later
                        optimizer.dual_step(current_constr, c_val=constr_ij)

                        # save the value of the constraint
                        c_log[-1].append(constr_ij.detach().numpy() - slack_vars[current_constr].detach().numpy() + threshold)

                        constraints[current_constr] = constr_ij

                        # iterate the constraint counter
                        current_constr += 1

            # calculate primal loss and grad
            loss = 0.0
            for i in range(0, number_of_constraints): # this is purely for pytorch not to complain about slack variables not being in the loss
                loss += 0*slack_vars[i]

            loss += criterion(out, batch_label)
            optimizer.step(loss, constraints)
            optimizer.zero_grad()

            # save the logs
            loss_log.append(loss.detach().numpy())
            duals_log.append(optimizer._dual_vars.detach())

            # slack variables must be non-negative. this is the "projection" step from the SSL-ALM paper
            with torch.no_grad():
                for s in slack_vars:
                    if s < 0:
                        s.zero_()

        SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSLALM_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints: {np.max(np.abs(np.mean(c_log, axis=0)))}/{np.max(np.abs(np.mean(t_c_log_plotting, axis=0)))}, "
        f"dual: {np.max(np.mean(duals_log, axis=0).mean())}"
        )

    return SSLALM_S_loss_log_plotting, SSLALM_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting

def pbm(seed_n, n_epochs, dataloader_train, dataloader_test, features_train, threshold):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 128
    hsize2 = 64
    model = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    # 5*6 constraint - per subgroup inequality + 2 per inequality
    number_of_constraints = 30
    criterion = torch.nn.BCEWithLogitsLoss()

    # optimizer = PBM(params=model_con.parameters(), m=number_of_constraints, lr=0.001, dual_beta=0.9, mu=0.1, penalty_update_m='CONST', barrier="quadratic_logarithmic", epoch_len=len(dataloader))
    optimizer = PBM(params=model.parameters(), m=number_of_constraints, lr=0.001, dual_beta=0.95, mu=0.1, 
                    penalty_update_m='DIMINISH', barrier="quadratic_logarithmic", epoch_len=len(dataloader_train))

    # alloc arrays for plotting
    PBM_S_loss_log_plotting = []  # mean
    PBM_S_c_log_plotting = []  # mean
    t_loss_log_plotting = []  # mean
    t_c_log_plotting = []  # mean

    t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)
    loss_log, c_log = test_model(model, dataloader_train, criterion, threshold)

    t_c_log_plotting.append(np.mean(t_c, axis=0))
    t_loss_log_plotting.append(np.mean(t_loss))

    PBM_S_c_log_plotting.append(np.mean(c_log, axis=0))
    PBM_S_loss_log_plotting.append(np.mean(loss_log))

    # training loop
    for epoch in range(n_epochs-1):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        a = 0
        for batch_input, batch_sens, batch_label in dataloader_train:

            # calculate constraints and constraint grads
            out = model(batch_input)

            # compute per group positive rate
            pos_rate_pergroup = positive_rate(out, batch_sens, prob_f=torch.nn.functional.sigmoid)

            # prepare counter + array of constr for this batch
            current_constr = 0
            c_log.append([])

            # compute the equal opportunity constraint
            for i in range(0, len(pos_rate_pergroup)):
                for j in range(0, len(pos_rate_pergroup)):

                    # calculate the constraint only for different subgroups
                    if i != j:

                        # the constraint with the slack variables
                        constr_ij = pos_rate_pergroup[i] - pos_rate_pergroup[j] 
                        constr_ij = constr_ij - threshold
                        
                        # save the value of the constraint
                        c_log[-1].append(constr_ij.detach().numpy() + threshold)

                        # perform the dual step variable + save the dual grad for later
                        optimizer.dual_step(current_constr, c_val=constr_ij)

                        # iterate the constraint counter
                        current_constr += 1

            # compute the augemented objective loss
            loss = criterion(out, batch_label)
            optimizer.step(loss)

            a += 1
            # save the logs
            loss_log.append(loss.detach().numpy())
            duals_log.append(optimizer._dual_vars.detach())

        PBM_S_c_log_plotting.append(np.mean(c_log, axis=0))
        PBM_S_loss_log_plotting.append(np.mean(loss_log))

        # test the model
        t_loss, t_c = test_model(model, dataloader_test, criterion, threshold)

        t_c_log_plotting.append(np.mean(t_c, axis=0))
        t_loss_log_plotting.append(np.mean(t_loss))

        print(
            f"Epoch: {epoch}, "
            f"loss Train/Test: {np.mean(loss_log)}/{np.mean(t_loss)}, "
            f"constraints: {np.max(np.abs(np.mean(c_log, axis=0)))}/{np.max(np.abs(np.mean(t_c_log_plotting, axis=0)))}, "
            f"dual: {np.max(np.mean(duals_log, axis=0).mean())}"
        )
    
    return PBM_S_loss_log_plotting, PBM_S_c_log_plotting, t_loss_log_plotting, t_c_log_plotting

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
    n_epochs = 30
    n_constraints = 30
    threshold = 0.1

    # define seeds
    seeds = [1, 2, 3]

    # log path file
    log_path = "./data/logs/log_benchmark_stochastic_2_bench.npz"

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
            "SSL-ALM",
            "SPBM"
        ],
        log_constraints=False,
        std_multiplier=1,
        mode='train_test', # change this to 'train', to ignore the test=
        plot_time_instead_epochs=False,
        save_path="./data/figs/ACSIncome_equal_opportunity_bench.pdf"
    )