import os

import folktables
import numpy as np
import pandas as pd
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

RAC1P_WHITE = 1


def download_folktables(
    state="AL",
    horizon="1-Year",
    survey="person",
    year=2018,
    download=True,
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


def prepare_folktables_multattr(
    dataset: str = "income",
    state="AL",
    horizon="1-Year",
    survey="person",
    year=2018,
    random_state=42,
    onehot=False,
    download=False,
    path=None,
    sens_cols=["RAC1P", "SEX"],
    binarize=[False, False],
    binarize_values=None,
    stratify=False,
    test_size=0.2,
    validation_size=0.5,
):
    acs_data, definition_df = download_folktables(
        state, horizon, survey, year, download, path
    )

    # ACSProblem = folktables.BasicProblem(
    #     features=ACSIncome.features,
    #     target="PINCP",
    #     target_transform=lambda x: x > 50000,
    #     group=sens_cols,
    #     group_transform=lambda x:
    #     preprocess=folktables.adult_filter,
    #     postprocess=lambda x: np.nan_to_num(x, -1),
    # )

    # if dataset == "employment":
    #     features, label, _ = ACSEmployment.df_to_numpy(acs_data)
    # elif dataset == "coverage":
    #     features, label, _ = ACSPublicCoverage.df_to_numpy(acs_data)
    if dataset == "income":
        categories = generate_categories(
            features=ACSIncome.features, definition_df=definition_df
        )
        features, label, _ = ACSIncome.df_to_pandas(acs_data)
        for col in categories.keys():
            features[col] = features[col].astype("category")

    for i, c in enumerate(sens_cols):
        if binarize[i]:
            features[c] = features[c].apply(lambda x: int(x == binarize[i]))

    # group membership (by combination of values of every sensitive attribute)
    sensitive_groups = features[sens_cols].apply(
        lambda x: "_".join([str(int(v)) for v in x[sens_cols]]), axis=1
    )

    sensitive_groups_onehot = torch.zeros(
        size=(len(features), len(sensitive_groups.unique()))
    )
    group_codes = []

    for gn, x in enumerate(sensitive_groups.unique()):
        group_codes.append(gn)
        sensitive_groups_onehot[sensitive_groups == x, gn] = 1.0

    # groups defined by sensitive attributes separately
    separate_sensitive_groups = [features[col].to_numpy() for col in sens_cols]

    if onehot:
        features_desens = pd.get_dummies(features.drop(sens_cols, axis=1)).to_numpy()
    else:
        features_desens = features.drop(sens_cols, axis=1).to_numpy()

    label = label.to_numpy().flatten()
    sensitive_groups = sensitive_groups.to_numpy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        group_train,
        group_test,
        group_onehot_train,
        group_onehot_test,
    ) = train_test_split(
        features_desens,
        label,
        sensitive_groups,
        sensitive_groups_onehot,
        test_size=test_size,
        stratify=sensitive_groups if stratify else None,
        random_state=random_state,
    )

    (
        X_val,
        X_test,
        y_val,
        y_test,
        group_val,
        group_test,
        group_onehot_val,
        group_onehot_test,
    ) = train_test_split(
        X_test,
        y_test,
        group_test,
        group_onehot_test,
        test_size=1 - validation_size,
        stratify=group_test if stratify else None,
        random_state=random_state,
    )

    # sep_sens_train = []
    # sep_sens_test = []
    # for gr in separate_sensitive_groups:
    #     s_train, s_test = train_test_split(
    #         gr,
    #         test_size=0.2,
    #         stratify=sensitive_groups if stratify else None,
    #         random_state=random_state,
    #     )
    #     sep_sens_train.append(
    #         [np.argwhere(s_train == sens_val).flatten() for sens_val in np.unique(gr)]
    #     )
    #     sep_sens_test.append(
    #         [np.argwhere(s_test == sens_val).flatten() for sens_val in np.unique(gr)]
    #     )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    group_indices_train = [
        np.argwhere(group_train == group_values).flatten()
        for group_values in np.unique(sensitive_groups)
    ]
    group_indices_val = [
        np.argwhere(group_val == group_values).flatten()
        for group_values in np.unique(sensitive_groups)
    ]
    group_indices_test = [
        np.argwhere(group_test == group_values).flatten()
        for group_values in np.unique(sensitive_groups)
    ]

    group_order = np.unique(sensitive_groups)

    return (
        (X_train_scaled, X_val_scaled, X_test_scaled),
        (y_train, y_val, y_test),
        (group_indices_train, group_indices_val, group_indices_test),
        (group_onehot_train, group_onehot_val, group_onehot_test),
        group_order,
        # group_indices_train,
        # group_onehot_train,
        # sep_sens_train,
        # X_test_scaled,
        # y_test,
        # group_indices_test,
        # sep_sens_test,
        # group_onehot_test,
    )
