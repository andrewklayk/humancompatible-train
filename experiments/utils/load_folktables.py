import os

import folktables
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSPublicCoverage,
    generate_categories,
)
import torch

# sys.path.append("..")


ACSIncomeSex = folktables.BasicProblem(
    features=ACSIncome.features,
    target="PINCP",
    target_transform=lambda x: x > 50000,
    group="SEX",
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, subset_indices, batch_size, drop_last=False):
        """
        Args:
            subset_indices (list of list): List of indices for each subset.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch.
        """
        self.subset_indices = subset_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_subsets = len(subset_indices)
        self.subset_sizes = [len(indices) for indices in subset_indices]
        self.n_samples_per_subset = batch_size // self.n_subsets

        # Check if batch_size is divisible by the number of subsets
        assert batch_size % self.n_subsets == 0, (
            f"Batch size ({batch_size}) must be divisible by the number of subsets ({self.n_subsets})."
        )

    def __iter__(self):
        # Shuffle indices within each subset
        shuffled_subset_indices = [torch.randperm(len(indices)).tolist() for indices in self.subset_indices]

        # Calculate the maximum number of batches per subset
        max_batches = min(len(indices) // self.n_samples_per_subset for indices in self.subset_indices)
        if not self.drop_last and any(len(indices) % self.n_samples_per_subset != 0 for indices in self.subset_indices):
            max_batches += 1  # Include partial batches if drop_last is False

        # Yield balanced batches
        for batch_idx in range(max_batches):
            batch = []
            for subset_idx in range(self.n_subsets):
                start = batch_idx * self.n_samples_per_subset
                end = start + self.n_samples_per_subset
                subset_batch_indices = shuffled_subset_indices[subset_idx][start:end]
                batch.extend([self.subset_indices[subset_idx][i] for i in subset_batch_indices])

            # Yield the global indices for the batch
            yield batch

    def __len__(self):
        if self.drop_last:
            return min(len(indices) // self.n_samples_per_subset for indices in self.subset_indices)
        else:
            return max((len(indices) + self.n_samples_per_subset - 1) // self.n_samples_per_subset
                       for indices in self.subset_indices)



RAC1P_WHITE = 1


def download_folktables(
    state="AL",
    horizon="1-Year",
    survey="person",
    year=2018,
    download=False,
    path=None,
):
    if path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "raw_data"))
    else:
        base_dir = path

    data_dir = os.path.join(base_dir, str(year), horizon)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    data_source = ACSDataSource(
        survey_year=year, horizon=horizon, survey=survey, root_dir=base_dir
    )

    definition_df = data_source.get_definitions(download=download)
    acs_data = data_source.get_data(states=[state], download=download)

    return acs_data, definition_df


def prepare_folktables(
    dataset: str = "income",
    state="AL",
    horizon="1-Year",
    survey="person",
    year=2018,
    random_state=None,
    onehot=True,
    download=False,
    path=None,
    sens_col="RAC1P",
    stratify=False,
):
    acs_data, definition_df = download_folktables(
        state, horizon, survey, year, download, path
    )

    # group here refers to race (RAC1P)
    if dataset == "employment":
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
        # drop the RAC1P column
        features = features[:, :-1]
    elif dataset == "coverage":
        features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    elif dataset == "income":
        if onehot:
            categories = generate_categories(
                features=ACSIncome.features, definition_df=definition_df
            )
            features, label, group = ACSIncome.df_to_pandas(
                acs_data, categories=categories, dummies=True
            )
            sens_features = [
                col for col in features.columns if col.startswith(sens_col)
            ]
            features = features.drop(columns=sens_features).to_numpy(dtype="float")
            label = label.to_numpy(dtype="float")
        if sens_col == "RAC1P":
            features, label, group = ACSIncome.df_to_pandas(acs_data)
        elif sens_col == "SEX":
            features, label, group = ACSIncomeSex.df_to_pandas(acs_data)

    features = features.drop(sens_col, axis=1)
    features = features.to_numpy()
    label = label.to_numpy().flatten()
    group = group.to_numpy()

    # drop sensitive
    group_binary = (group == RAC1P_WHITE).astype(float)

    # stratify by binary race (white vs rest)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        features,
        label,
        group_binary,
        test_size=0.2,
        stratify=group_binary if stratify else None,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_w_idx = np.argwhere(g_train == 1).flatten()
    train_nw_idx = np.argwhere(g_train != 1).flatten()

    test_w_idx = np.argwhere(g_test == 1).flatten()
    test_nw_idx = np.argwhere(g_test != 1).flatten()

    return (
        X_train_scaled,
        y_train,
        [train_w_idx, train_nw_idx],
        X_test_scaled,
        y_test,
        [test_w_idx, test_nw_idx],
    )


def prepare_folktables_multattr(
    dataset: str = "income",
    state="AL",
    horizon="1-Year",
    survey="person",
    year=2018,
    random_state=None,
    onehot=True,
    download=False,
    path=None,
    sens_cols=["RAC1P", "SEX"],
    binarize=[False, False],
    binarize_values=None,
    stratify=False,
):
    acs_data, definition_df = download_folktables(
        state, horizon, survey, year, download, path
    )

    if dataset == "employment":
        features, label, group = ACSEmployment.df_to_numpy(acs_data)
    elif dataset == "coverage":
        features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)
    elif dataset == "income":
        features, label, _ = ACSIncome.df_to_pandas(acs_data)

    for i, c in enumerate(sens_cols):
        if binarize[i]:
            features[c] = features[c].apply(lambda x: int(x == binarize[i]))

    # group membership (by combination of values of every sensitive attribute)
    sensitive_groups = features[sens_cols].apply(
        lambda x: "_".join([str(int(v)) for i, v in enumerate(x[sens_cols])]), axis=1
    )

    sensitive_groups_onehot = torch.zeros(size=(len(features), len(sensitive_groups.unique())))
    group_codes = []

    for gn, x in enumerate(sensitive_groups.unique()):
        group_codes.append(gn)
        sensitive_groups_onehot[sensitive_groups == x, gn] = 1.

    # group_codes = {
    #     {c[] for c in sens_cols}
    # }

    # groups defined by sensitive attributes separately
    separate_sensitive_groups = [features[col].to_numpy() for col in sens_cols]

    features_desens = features.drop(sens_cols, axis=1).to_numpy()
    label = label.to_numpy().flatten()
    sensitive_groups = sensitive_groups.to_numpy()

    X_train, X_test, y_train, y_test, group_train, group_test, group_onehot_train, group_onehot_test = train_test_split(
        features_desens,
        label,
        sensitive_groups,
        sensitive_groups_onehot,
        test_size=0.2,
        stratify=sensitive_groups if stratify else None,
        random_state=random_state,
    )

    sep_sens_train = []
    sep_sens_test = []
    for gr in separate_sensitive_groups:
        s_train, s_test = train_test_split(
            gr,
            test_size=0.2,
            stratify=sensitive_groups if stratify else None,
            random_state=random_state,
        )
        sep_sens_train.append(
            [np.argwhere(s_train == sens_val).flatten() for sens_val in np.unique(gr)]
        )
        sep_sens_test.append(
            [np.argwhere(s_test == sens_val).flatten() for sens_val in np.unique(gr)]
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    group_indices_train = [
        np.argwhere(group_train == sens_val).flatten()
        for sens_val in np.unique(sensitive_groups)
    ]
    group_indices_test = [
        np.argwhere(group_test == sens_val).flatten()
        for sens_val in np.unique(sensitive_groups)
    ]

    group_order = np.unique(sensitive_groups)
    # separate_group_indices = [separate_sensitive_groups

    return (
        X_train_scaled,
        y_train,
        group_indices_train,
        group_onehot_train,
        sep_sens_train,
        X_test_scaled,
        y_test,
        group_indices_test,
        sep_sens_test,
        group_onehot_test,
        group_order
    )