import os
from scipy.io.arff import loadarff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

RAC1P_WHITE = 1


def load_dutch(
    path=None,
):
    if path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "raw_data"))
    else:
        base_dir = path

    raw_data = loadarff('utils/raw_data/dutch_census_2001.arff')
    df_data = pd.DataFrame(raw_data[0])

    return df_data

def prepare_dutch(
        onehot=False,
        sens_cols=["sex"],
        stratify=False,
        random_state=None,
        test_size=0.2,
        validation_size=0.5
):
    df = load_dutch()
    features = df.drop(['occupation'], axis=1)
    labels = df['occupation']
    sensitive_groups = features[sens_cols].to_numpy()
    sensitive_groups_onehot = torch.zeros(size=(len(features), len(sensitive_groups.unique())))
    group_codes = []

    for gn, x in enumerate(sensitive_groups.unique()):
        group_codes.append(gn)
        sensitive_groups_onehot[sensitive_groups == x, gn] = 1.

    # groups defined by sensitive attributes separately
    separate_sensitive_groups = [features[col].to_numpy() for col in sens_cols]
    
    if onehot:
        features_desens = pd.get_dummies(features.drop(sens_cols, axis=1)).to_numpy()
    else:
        features_desens = features.drop(sens_cols, axis=1).to_numpy()

    labels = labels.to_numpy().flatten()
    sensitive_groups = sensitive_groups.to_numpy()

    X_train, X_test, y_train, y_test, group_train, group_test, group_onehot_train, group_onehot_test = train_test_split(
        features_desens,
        labels,
        sensitive_groups,
        sensitive_groups_onehot,
        test_size=test_size,
        stratify=sensitive_groups if stratify else None,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test, group_val, group_test, group_onehot_val, group_onehot_test = train_test_split(
        X_test,
        y_test,
        group_test,
        group_onehot_test,
        test_size=1-validation_size,
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

    return(
        (X_train_scaled, X_val_scaled, X_test_scaled),
        (y_train, y_val, y_test),
        (group_indices_train, group_indices_val, group_indices_test),
        (group_onehot_train, group_onehot_val, group_onehot_test),
        group_order
        # group_indices_train,
        # group_onehot_train,
        # sep_sens_train,
        # X_test_scaled,
        # y_test,
        # group_indices_test,
        # sep_sens_test,
        # group_onehot_test,
    )