from humancompatible.train.optim.ssw_barrier import SSG_Barrier
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

def test_ssw_barrier_deterministic():

    # define the torch seed here
    seed_n = 1
    n_epochs = 10

    # log path file
    log_path = "./data/logs/log_benchmark.npz"

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
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

    # set the seed for fair comparisons
    torch.manual_seed(seed_n)

    # create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)

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
    optimizer = SSG_Barrier(params=model.parameters(), m=1, lr=0.01, dual_lr=0.1, obj_lr_infeas=0.01)

    # bounds for the constraints: norm of max each weight matrix should be <= 1
    constraint_bounds = [1.0]

    # define epochs + loss function - same loss should be defined for all algorithms
    criterion = torch.nn.BCEWithLogitsLoss()

    # alloc the plotting array
    SSG_c_log_plotting = []
    SSG_loss_log_plotting = []
    SSG_std_loss_log_plotting = []

    # training loop
    for epoch in range(n_epochs):
        # alloc logging array
        loss_log = []
        c_log = []

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

            # save the gradient of the constraint
            optimizer.dual_step(0, max_norm_viol)
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
        SSG_std_loss_log_plotting.append(np.std(loss_log))

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )


def test_ssw_barrier_stochastic():

    # define the torch seed here
    seed_n = 1
    n_epochs = 10

    # log path file
    log_path = "./data/logs/log_benchmark.npz"

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
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

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
    optimizer = SSG_Barrier(params=model.parameters(), m=1, lr=0.1, dual_lr=0.01, obj_lr_infeas=0.001)

    # define the criterion
    criterion = torch.nn.BCEWithLogitsLoss()
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    fair_crit_bound = 0.2

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
            # fair_constraint.backward(retain_graph=True)

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

        print(
            f"Epoch: {epoch}, "
            f"loss: {np.mean(loss_log)}, "
            f"constraints: {np.mean(c_log, axis=0)}, "
        )

if __name__ == "__main__":

    # test_ssw_barrier_deterministic()
    test_ssw_barrier_stochastic()