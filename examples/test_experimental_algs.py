from humancompatible.train.optim.PBM import PBM
from humancompatible.train.optim.ssw_barrier import SSG_Adam
from humancompatible.train.optim.ssw import SSG
# load and prepare data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from folktables import ACSDataSource, ACSIncome, generate_categories
from torch.nn import Sequential
from fairret.statistic import PositiveRate
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import numpy as np
import matplotlib.pyplot as plt
from humancompatible.train.optim.ssl_alm_adam_moment import SSLALM_Adam_moment
from humancompatible.train.optim.ssl_alm_adam import SSLALM_Adam

def plot_losses_and_constraints_single_stochastic(
    losses_list,
    losses_std_list,
    constraints_list,
    constraints_std_list,
    constraint_thresholds,
    titles=None,
    eval_points=2,
    std_multiplier=2,
    log_constraints=False,
    savepath=None
):
    # --- Color palette: Tableau 10 ---
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

    # --- Marker styles (reused from inspired function) ---
    marker_styles = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
    marker_styles = (marker_styles * ((len(losses_list) // len(marker_styles)) + 1))[
        : len(losses_list)
    ]

    num_algos = len(losses_list)
    if titles is None:
        titles = [f"Algorithm {i + 1}" for i in range(num_algos)]
    constraint_thresholds = np.atleast_1d(constraint_thresholds)

    fig, axes = plt.subplots(2, 1, figsize=(9, 11))
    ax_loss, ax_constr = axes

    # --- LOSS PLOT ---
    for j, (loss, loss_std) in enumerate(zip(losses_list, losses_std_list)):
        x = np.arange(len(loss))
        color = colors[j % len(colors)]
        upper = loss + std_multiplier * loss_std
        lower = loss - std_multiplier * loss_std

        # Mean curve
        ax_loss.plot(x, loss, lw=2.2, color=color, label=titles[j])
        # Std shading
        ax_loss.fill_between(x, lower, upper, color=color, alpha=0.15)

        # Eval points
        if eval_points is not None:
            if isinstance(eval_points, int):
                idx = np.arange(0, len(loss), eval_points)
            else:
                idx = np.array(eval_points)
                idx = idx[idx < len(loss)]
            ax_loss.plot(
                x[idx],
                loss[idx],
                marker_styles[j],
                color=color,
                markersize=6,
                alpha=0.8,
            )

    ax_loss.set_ylabel("Mean Loss")
    ax_loss.set_title("Loss Comparison")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.legend(fontsize=9)

    # --- CONSTRAINT PLOT ---
    for j, (constraints, constraints_std) in enumerate(
        zip(constraints_list, constraints_std_list)
    ):
        color = colors[j % len(colors)]
        constraints = np.array(constraints)
        constraints_std = np.array(constraints_std)
        x = np.arange(constraints.shape[1])

        c_min = np.min(constraints - std_multiplier * constraints_std, axis=0)
        c_max = np.max(constraints + std_multiplier * constraints_std, axis=0)

        # Fill min-max range
        ax_constr.fill_between(
            x, c_min, c_max, color=color, alpha=0.15, label=titles[j]
        )

        # Plot mean curves with markers
        for c_mean in constraints:
            ax_constr.plot(x, c_mean, lw=1.8, color=color, alpha=0.7)
            if eval_points is not None:
                if isinstance(eval_points, int):
                    idx = np.arange(0, len(c_mean), eval_points)
                else:
                    idx = np.array(eval_points)
                    idx = idx[idx < len(c_mean)]
                ax_constr.plot(
                    x[idx],
                    c_mean[idx],
                    marker_styles[j],
                    color=color,
                    markersize=5,
                    alpha=0.8,
                )

    # Threshold lines
    for th in constraint_thresholds:
        y = np.log(th) if log_constraints else th
        ax_constr.axhline(y, color="red", linestyle="--", lw=1.4, label="Threshold")

    ax_constr.set_ylabel("Log Constraint" if log_constraints else "Constraint")
    ax_constr.set_xlabel("Epoch")
    ax_constr.set_title("Constraint Comparison")
    ax_constr.grid(True, linestyle="--", alpha=0.35)
    ax_constr.legend(fontsize=9)

    plt.tight_layout()
    if savepath is None: 
        plt.show()
    else: 
        plt.savefig(savepath, dpi=100)


def test_ssw_adam_stochastic(log_path_save, n_epochs, seed_n):

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


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
    optimizer = SSG_Adam(params=model.parameters(), m=1, lr=0.01, dual_lr=0.005, obj_lr_infeas=0.001)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean
    SSG_S_loss_std_log_plotting = []  # std
    SSG_S_c_std_log_plotting = []  # std

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)
            fair_constraint = torch.max(fair_loss - fair_crit_bound, torch.zeros(1))
        
            # compute the grad of the constraints
            optimizer.dual_step(0, fair_constraint)
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
        SSG_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSG_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )

    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSG_S_loss_log_plotting)]
    constraints += [np.array(SSG_S_c_log_plotting).T]
    losses_std += [np.array(SSG_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSG_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path_save,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )

    print("\nSSG done!\n")


def test_ssw_stochastic(log_path_save, n_epochs, seed_n):

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


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
    
    # create the SSLALM optimizer
    optimizer = SSG(params=model.parameters(), m=1, lr=0.1, dual_lr=0.05)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean
    SSG_S_loss_std_log_plotting = []  # std
    SSG_S_c_std_log_plotting = []  # std

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:
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
        SSG_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSG_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )

    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSG_S_loss_log_plotting)]
    constraints += [np.array(SSG_S_c_log_plotting).T]
    losses_std += [np.array(SSG_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSG_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path_save,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )

    print("\nSSG done!\n")
    

def test_PBM_barrier_stochastic(log_path_save, n_epochs, seed_n):

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


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
    # optimizer = PBM(params=model.parameters(), m=1, lr=0.01, dual_lr=0.1, dual_beta=0.999, p=0.1, barrier="augmented_lagrangian")
    # optimizer = PBM(params=model.parameters(), m=1, lr=0.01, rho=1.0, init_pi=10_000, dual_lr=0.1, dual_beta=0.999, penalty_update_m='ALM', barrier="quadratic_logarithmic")
    # optimizer = PBM(params=model.parameters(), m=1, lr=0.005, dual_beta=0.95, penalty_update_m='CONST', barrier="quadratic_logarithmic")
    optimizer = PBM(params=model.parameters(), m=1, lr=0.001, dual_beta=0.95, mu=0.1, penalty_update_m='CONST', barrier="quadratic_logarithmic")

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean
    SSG_S_loss_std_log_plotting = []  # std
    SSG_S_c_std_log_plotting = []  # std

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)
            fair_constraint = fair_loss - fair_crit_bound

            # compute the grad of the constraints
            optimizer.dual_step(0, fair_constraint)
            optimizer.zero_grad()

            # save the constraint value
            c_log.append([fair_loss.detach().item()])
            duals_log.append(optimizer._dual_vars.detach())

            # calculate loss and grad
            loss = criterion(out, batch_label)
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

        SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSG_S_loss_log_plotting.append(np.mean(loss_log))
        SSG_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSG_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )

    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSG_S_loss_log_plotting)]
    constraints += [np.array(SSG_S_c_log_plotting).T]
    losses_std += [np.array(SSG_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSG_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path_save,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )
    
    print("\nPBM done!\n")

def test_unconstraint_stochastic(log_path_save, n_epochs, seed_n):

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)


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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2

    # alloc arrays for plotting
    SSG_S_loss_log_plotting = []  # mean
    SSG_S_c_log_plotting = []  # mean
    SSG_S_loss_std_log_plotting = []  # std
    SSG_S_c_std_log_plotting = []  # std

    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:
            # calculate constraints and constraint grads
            out = model(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # save the constraint value
            c_log.append([fair_loss.detach().item()])

            # calculate loss and grad
            loss = criterion(out, batch_label)
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

        SSG_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSG_S_loss_log_plotting.append(np.mean(loss_log))
        SSG_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSG_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )

    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSG_S_loss_log_plotting)]
    constraints += [np.array(SSG_S_c_log_plotting).T]
    losses_std += [np.array(SSG_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSG_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path_save,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )
    
    print("\nUnconstraint done!\n")

def test_sslalm_dualmoment_stochastic(log_path_save, n_epochs, seed_n):
    
    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )
    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2  # define the bound on the criterion

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 64
    hsize2 = 32
    model_con = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    optimizer = SSLALM_Adam_moment(
        params=model_con.parameters(),
        m=1,  # number of constraints - one in our case
        lr=0.01,  # primal variable lr
        dual_lr=0.005,  # lr of a dual ALM variable
        dual_bound=5,
        rho=1,  # rho penalty in ALM parameter
        mu=2,  # smoothing parameter
    )

    # add slack variables - to create the equality from the inequalities    
    slack_vars = torch.zeros(1, requires_grad=True)
    optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})

    # alloc arrays for plotting
    SSLALM_S_loss_log_plotting = []  # mean
    SSLALM_S_c_log_plotting = []  # mean
    SSLALM_S_loss_std_log_plotting = []  # std
    SSLALM_S_c_std_log_plotting = []  # std


    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:

            # calculate constraints and constraint grads
            out = model_con(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # calculate the fair constraint violation
            fair_constraint = fair_loss + slack_vars[0] - fair_crit_bound
            fair_constraint.backward(retain_graph=True)

            # perform the dual step variable + save the dual grad for later
            optimizer.dual_step(0, c_val=fair_constraint)
            optimizer.zero_grad()

            # save the fair loss violation for logging
            c_log.append([fair_loss.detach().item()])
            duals_log.append(optimizer._dual_vars.detach())

            # calculate primal loss and grad
            loss = criterion(out, batch_label) + 0 * slack_vars[0]
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

            # slack variables must be non-negative. this is the "projection" step from the SSL-ALM paper
            with torch.no_grad():
                for s in slack_vars:
                    if s < 0:
                        s.zero_()

        # optimizer.dual_lr *= 0.95
        SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSLALM_S_loss_log_plotting.append(np.mean(loss_log))
        SSLALM_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSLALM_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )


    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSLALM_S_loss_log_plotting)]
    constraints += [np.array(SSLALM_S_c_log_plotting).T]
    losses_std += [np.array(SSLALM_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSLALM_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )


def test_sslalm_stochastic(log_path_save, n_epochs, seed_n):

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

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

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=128, drop_last=True
    )
    # create a dataloader from the sampler
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2  # define the bound on the criterion

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    hsize1 = 64
    hsize2 = 32
    model_con = Sequential(
        torch.nn.Linear(features_train.shape[1], hsize1),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize1, hsize2),
        torch.nn.ReLU(),
        torch.nn.Linear(hsize2, 1),
    )

    optimizer = SSLALM_Adam(
        params=model_con.parameters(),
        m=1,  # number of constraints - one in our case
        lr=0.01,  # primal variable lr
        dual_lr=0.02,  # lr of a dual ALM variable
        dual_bound=5,
        rho=1,  # rho penalty in ALM parameter
        mu=0.1,  # smoothing parameter
    )

    # add slack variables - to create the equality from the inequalities    
    slack_vars = torch.zeros(1, requires_grad=True)
    optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})

    # alloc arrays for plotting
    SSLALM_S_loss_log_plotting = []  # mean
    SSLALM_S_c_log_plotting = []  # mean
    SSLALM_S_loss_std_log_plotting = []  # std
    SSLALM_S_c_std_log_plotting = []  # std


    # training loop
    for epoch in range(n_epochs):
        # alloc the logging arrays for the batch
        loss_log = []
        c_log = []
        duals_log = []

        # go though all data
        for batch_input, batch_sens, batch_label in dataloader:

            # calculate constraints and constraint grads
            out = model_con(batch_input)
            fair_loss = fair_criterion(out, batch_sens)

            # calculate the fair constraint violation
            fair_constraint = fair_loss + slack_vars[0] - fair_crit_bound
            fair_constraint.backward(retain_graph=True)

            # perform the dual step variable + save the dual grad for later
            optimizer.dual_step(0, c_val=fair_constraint)
            optimizer.zero_grad()

            # save the fair loss violation for logging
            c_log.append([fair_loss.detach().item()])
            duals_log.append(optimizer._dual_vars.detach())

            # calculate primal loss and grad
            loss = criterion(out, batch_label) + 0 * slack_vars[0]
            loss.backward()
            loss_log.append(loss.detach().numpy())
            optimizer.step()
            optimizer.zero_grad()

            # slack variables must be non-negative. this is the "projection" step from the SSL-ALM paper
            with torch.no_grad():
                for s in slack_vars:
                    if s < 0:
                        s.zero_()

        # optimizer.dual_lr *= 0.95
        SSLALM_S_c_log_plotting.append(np.mean(c_log, axis=0))
        SSLALM_S_loss_log_plotting.append(np.mean(loss_log))
        SSLALM_S_c_std_log_plotting.append(np.std(c_log, axis=0))
        SSLALM_S_loss_std_log_plotting.append(np.std(loss_log, axis=0))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
            f"dual: {np.mean(duals_log, axis=0)}"
        )


    log_path = log_path_save
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    # append
    losses += [np.array(SSLALM_S_loss_log_plotting)]
    constraints += [np.array(SSLALM_S_c_log_plotting).T]
    losses_std += [np.array(SSLALM_S_loss_std_log_plotting)]
    constraints_std += [np.array(SSLALM_S_c_std_log_plotting).T]

    # save the computed data
    np.savez(
        log_path,
        losses=losses,
        constraints=constraints,
        losses_std=losses_std,
        constraints_std=constraints_std,
    )
    

if __name__ == "__main__":

    savepath = "./examples/data/logs/testing_algs2.npz"
    epochs = 70
    seed = 1

    # save the computed data
    np.savez(
        savepath,
        losses=[],
        constraints=[],
        losses_std=[],
        constraints_std=[],
    )

    # train the old algorithms 
    print("Testing SSW: ")
    # test_ssw_stochastic(savepath, epochs, seed)
    
    print("Testing SSLALM: ")
    # test_sslalm_stochastic(savepath, epochs, seed)
    
    # # train the experimental algorithms
    # print("Testing SSW-ADAM: ")
    # test_ssw_adam_stochastic(savepath, epochs, seed)
    
    # print("Testing SSLALM-DM: ")
    # test_sslalm_dualmoment_stochastic(savepath, epochs, seed)

    print("Testing PBM: ")
    # test_PBM_barrier_stochastic(savepath, epochs, seed) 

    # print("Testing Unconstraint: ")
    # test_unconstraint_stochastic(savepath, epochs, seed)

    # plot the results
    log_path = savepath
    # load the prior and append
    losses = list(np.load(log_path)["losses"])
    constraints = list(np.load(log_path)["constraints"])
    losses_std = list(np.load(log_path)["losses_std"])
    constraints_std = list(np.load(log_path)["constraints_std"])

    thresholds = [0.2]
    plot_losses_and_constraints_single_stochastic(
    losses,
    losses_std,
    constraints,
    constraints_std,
    thresholds,
    titles=[
        "SSW",
        "SSLALM",
        "SSW-ADAM",
        "SSLALM-DM",
        "PBM",
        "Unconstraint"
    ],
    log_constraints=False,
    std_multiplier=1,
    savepath="./examples/data/figs/experimental_algs2.png"
    )