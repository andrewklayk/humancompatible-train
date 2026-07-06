"""Dataset loaders with K-fold cross-validation and separated randomness.

Two independent randomness sources (see run.py / README):
  * ``cv_seed``   -- the data partition only: a single fixed held-out TEST set
    (stratified) is split off BEFORE K-folding, then the remaining *dev* set is
    partitioned with StratifiedKFold(n_splits=n_folds, shuffle=True,
    random_state=cv_seed). ``fold`` selects which fold is validation.
  * ``init_seed`` -- the optimization side: the BalancedBatchSampler / DataLoader
    generator (batch order). Model init is seeded in run.py, also from init_seed.

Stratification is by sensitive group (class for cifar) so every group appears in
the test set and in every fold -- required by BalancedBatchSampler and the
per-group constraints. The StandardScaler is fit on each fold's TRAIN split only
(no leakage from val/test).

Dataset-specific preprocessing (folktables download, dutch transform, cifar) is
unchanged; only the split/scale/loader tail is shared via ``_cv_load``.
"""
import itertools
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, generate_categories, ACSIncome, BasicProblem
from humancompatible.train.fairness.utils import BalancedBatchSampler


def _stratify_key(groups):
    """1-D class label per row for stratification (argmax of a one-hot)."""
    groups = np.asarray(groups)
    return groups.argmax(1) if groups.ndim > 1 else groups.astype(int)


def _cv_indices(n, strat, cv_seed, n_folds, fold, test_size):
    """Fixed stratified test hold-out, then StratifiedKFold over the dev set.

    Returns (train_idx, val_idx, test_idx) into [0, n). The test set and the fold
    partition depend ONLY on cv_seed/n_folds/test_size, never on init_seed.
    """
    idx = np.arange(n)
    dev_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=cv_seed, stratify=strat)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    try:
        tr_rel, val_rel = list(skf.split(dev_idx, strat[dev_idx]))[fold]
    except ValueError as e:
        raise ValueError(
            f"StratifiedKFold(n_splits={n_folds}) failed -- a group likely has "
            f"fewer than {n_folds} members in the dev set. Reduce n_folds or drop "
            f"small groups. Original error: {e}")
    return dev_idx[tr_rel], dev_idx[val_rel], test_idx


def _cv_load(features, groups, labels, *, batch_size, device, cv_seed, n_folds, fold,
             init_seed, test_size, dtype=torch.float32, balanced=True,
             extend_groups=False, val_test_batch=None, approach="opt"):
    """Shared tabular tail: stratified test hold-out + K-fold dev split + loaders.

    Returns the standard 4-tuple
        ((train_loader, val_loader, test_loader),
         (f,g,l)_train, (f,g,l)_val, (f,g,l)_test).
    val/test entries are None when approach='opt' (full dataset, train only).
    """
    features = np.asarray(features)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    strat = _stratify_key(groups)

    if approach == "opt":
        scaler = StandardScaler()
        X_all = scaler.fit_transform(features)
        def T(a):
            return torch.tensor(a).to(dtype).to(device)
        ftr, gtr, ytr = T(X_all), T(groups), T(labels)
        ds_tr = torch.utils.data.TensorDataset(ftr, gtr, ytr)
        g = torch.Generator(device=device)
        g.manual_seed(init_seed)
        if balanced:
            eg = list(range(gtr.shape[1])) if extend_groups else None
            sampler = BalancedBatchSampler(group_onehot=gtr, batch_size=batch_size,
                                           drop_last=True, extend_groups=eg, generator=g)
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_sampler=sampler, generator=g)
        else:
            dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)
        return (dl_tr, None, None), (ftr, gtr, ytr), None, None

    train_idx, val_idx, test_idx = _cv_indices(
        len(features), strat, cv_seed, n_folds, fold, test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_val = scaler.transform(features[val_idx])
    X_test = scaler.transform(features[test_idx])

    def T(a):
        return torch.tensor(a).to(dtype).to(device)

    ftr, gtr, ytr = T(X_train), T(groups[train_idx]), T(labels[train_idx])
    fval, gval, yval = T(X_val), T(groups[val_idx]), T(labels[val_idx])
    fte, gte, yte = T(X_test), T(groups[test_idx]), T(labels[test_idx])

    ds_tr = torch.utils.data.TensorDataset(ftr, gtr, ytr)
    ds_val = torch.utils.data.TensorDataset(fval, gval, yval)
    ds_te = torch.utils.data.TensorDataset(fte, gte, yte)

    # Batch order is the init-side randomness.
    g = torch.Generator(device=device)
    g.manual_seed(init_seed)
    vtb = val_test_batch or batch_size
    if balanced:
        eg = list(range(gtr.shape[1])) if extend_groups else None
        sampler = BalancedBatchSampler(group_onehot=gtr, batch_size=batch_size,
                                       drop_last=True, extend_groups=eg, generator=g)
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_sampler=sampler, generator=g)
    else:
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)
    # val/test loaders are plain full passes (unused for tabular; used by cifar path
    # which has its own builder). shuffle off, keep all samples.
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=vtb, shuffle=False, generator=g)
    dl_te = torch.utils.data.DataLoader(ds_te, batch_size=vtb, shuffle=False, generator=g)

    return ((dl_tr, dl_val, dl_te), (ftr, gtr, ytr), (fval, gval, yval), (fte, gte, yte))


def comb_cat_dummies(df):
    # Group columns by prefix
    groups = {}
    for col in df.columns:
        prefix = col.split('_')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(col)

    # Generate all possible combinations of column names
    combination_columns = []
    for cols in itertools.product(*groups.values()):
        combo = '&'.join(cols)
        combination_columns.append(combo)

    # Create new columns for each combination
    for combo in combination_columns:
        cols = combo.split('&')
        df[combo] = df[cols].min(axis=1)

    # Drop the original columns if desired
    df = df[[col for col in df.columns if col not in [c for group in groups.values() for c in group]]]

    return df


def load_data_norm(batch_size, device, *, cv_seed, n_folds, fold, init_seed, test_size=0.2, approach="opt"):
    # load folktables data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["FL"], download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSIncome.features, definition_df=definition_df
    )
    df_feat, df_labels, _ = ACSIncome.df_to_pandas(
        acs_data, categories=categories, dummies=True
    )

    sens_cols = ["SEX_Female", "SEX_Male"]
    features = df_feat.drop(columns=sens_cols).to_numpy(dtype='float32')
    groups = df_feat[sens_cols].to_numpy(dtype='float32')
    labels = df_labels.to_numpy(dtype='float32')

    return _cv_load(features, groups, labels, batch_size=batch_size, device=device,
                    cv_seed=cv_seed, n_folds=n_folds, fold=fold, init_seed=init_seed,
                    test_size=test_size, balanced=False, approach=approach)


def load_data_FT(batch_size, device, sens_attrs, states=['FL'], group_size_threshold=0,
                 sens_groups=None, extend_groups=False, dtype=torch.float32,
                 *, cv_seed, n_folds, fold, init_seed, test_size=0.2, approach="opt"):
    # load folktables data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    ACSProblem = BasicProblem(
        features=ACSIncome.features,
        target=ACSIncome.target,
        target_transform=ACSIncome.target_transform,
        group=sens_attrs,
        group_transform=lambda x: pd.get_dummies(x, columns=sens_attrs),
        preprocess=ACSIncome._preprocess,
        postprocess=ACSIncome._postprocess
    )
    acs_data = data_source.get_data(states=states, download=True)
    definition_df = data_source.get_definitions(download=True)
    categories = generate_categories(
        features=ACSProblem.features, definition_df=definition_df
    )
    df_feat, df_labels, df_sens = ACSProblem.df_to_pandas(
        acs_data, categories=categories, dummies=True
    )

    if 'MAR' in sens_attrs:
        df_sens['MAR_2'] = df_sens['MAR_2'] + df_sens['MAR_4'] + df_sens['MAR_5']
        df_sens.drop(['MAR_4', 'MAR_5'], inplace=True, axis='columns')

    df_sens_onehot = comb_cat_dummies(df_sens) if len(sens_attrs) > 1 else df_sens

    features = df_feat.drop(columns=[col for col in df_feat.columns if col.startswith(tuple(sens_attrs))]).to_numpy(dtype='float')
    groups = df_sens_onehot.to_numpy(dtype='float')
    labels = df_labels.to_numpy(dtype='float')

    if sens_groups:
        group_names = []
        for group in sens_groups:
            group_def = []
            for key, val in group.items():
                group_def.append("_".join([str(key), str(val)]))
            group_str = "&".join(group_def)
            group_names.append(group_str)

        keep_mask = df_sens_onehot[group_names].any(axis=1)
        features = features[keep_mask]
        labels = labels[keep_mask]
        df_sens_onehot = df_sens_onehot[keep_mask].drop(columns=[col for col in df_sens_onehot if col not in group_names])
        groups = df_sens_onehot.to_numpy()

    for idx in range(groups.shape[1]):
        print(f"{df_sens_onehot.columns[idx]}, : {(groups[:, idx] == 1).sum()}")

    return _cv_load(features, groups, labels, batch_size=batch_size, device=device,
                    cv_seed=cv_seed, n_folds=n_folds, fold=fold, init_seed=init_seed,
                    test_size=test_size, dtype=dtype, balanced=True,
                    extend_groups=extend_groups, approach=approach)


def load_data_DUTCH(batch_size, device='cpu', extend_groups=False,
                    *, cv_seed, n_folds, fold, init_seed, test_size=0.4, approach="opt"):
    features, groups, labels, _ = get_data_dutch(drop_small_groups=True, print_stats=True)
    return _cv_load(features, groups, labels, batch_size=batch_size, device=device,
                    cv_seed=cv_seed, n_folds=n_folds, fold=fold, init_seed=init_seed,
                    test_size=test_size, dtype=torch.float32, balanced=True,
                    extend_groups=extend_groups, approach=approach)


def get_data_dutch(drop_small_groups=True, print_stats=True):
    """Loads the dutch dataset and returns FULL (unsplit, unscaled) arrays:
    (features[np float], groups_onehot[np], labels[np (N,1)], group_dict).
    Splitting + scaling are done per fold in ``_cv_load``."""
    from fairml_datasets import Dataset

    dataset = Dataset.from_id("dutch")
    df = dataset.load()

    if drop_small_groups:
        df = df.drop(df[(df.age == '13') | (df.age == '14') | (df.age == '15')].index)
        num_age_groups = 9
    else:
        num_age_groups = 12

    target_column = dataset.get_target_column()
    df_transformed, transformation_info = dataset.transform(df)
    sensitive_columns = ['age', 'sex_1', 'sex_2']

    df_labels = df_transformed[target_column]
    df_features = df_transformed.drop(columns=target_column)
    df_features = df_features.drop(columns=sensitive_columns)

    sex_cols = ['sex_1', 'sex_2']
    sex_idx = df_transformed[sex_cols].values.argmax(axis=1).astype(int)
    age = df_transformed['age'].values.astype(int)

    num_groups = num_age_groups * 2
    group_indices = sex_idx * num_age_groups + (age - 4)
    groups_onehot = np.eye(num_groups)[group_indices]

    group_dict = {}
    for i, (s, m) in enumerate(product(sex_cols, np.array(range(0, num_age_groups)) + 4)):
        group_dict[i] = f"{s} + age_{m}"

    if print_stats:
        print("Number of Samples per group: \n")
        for idx in group_dict:
            print(f"{group_dict[idx]}, : {(groups_onehot[:, idx] == 1).sum()}")

    features = df_features.to_numpy(dtype='float32')
    labels = np.asarray(df_labels, dtype='float32').reshape((-1, 1))
    return features, groups_onehot, labels, group_dict


def _balanced_subsample(X, targets, eye, num_classes, size, seed, device):
    """Fixed class-balanced subsample (size//num_classes per class), drawn with a
    dedicated ``seed`` so the optimality-eval set is identical across init seeds and
    epochs. Returns an (X, sens_onehot, targets) tuple. CIFAR is class-balanced, so
    a large balanced subsample is representative of the full empirical problem."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    per = max(1, size // num_classes)
    idxs = []
    for k in range(num_classes):
        cls_idx = (targets == k).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_idx), generator=g, device=device)[:per]
        idxs.append(cls_idx[perm])
    idx = torch.cat(idxs)
    return X[idx], eye[targets[idx]], targets[idx]


def load_data_cifar(num_classes, *, cv_seed, n_folds, fold, init_seed,
                    balanced=False, device='cpu', approach="opt", opt_eval_size=10000):
    """CIFAR-10 / CIFAR-100 with K-fold over the training set; the canonical
    torchvision test set is the fixed held-out TEST. ``sens`` is the class one-hot
    (for the "equal loss across classes" constraint). Stratified by class.

    Returns ``(trainloader, valloader, testloader, opt_eval)``. With approach='opt'
    the full training set is used (val/test None) and ``opt_eval`` is a fixed
    class-balanced subsample (<= ``opt_eval_size``) for the end-of-epoch optimality
    eval; otherwise ``opt_eval`` is None."""
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 120 if num_classes == 10 else 200

    ds_cls = torchvision.datasets.CIFAR10 if num_classes == 10 else torchvision.datasets.CIFAR100
    trainset = ds_cls(root='./data', train=True, download=True, transform=transform)

    X = torch.stack([item[0] for item in trainset]).to(device)
    targets = torch.tensor([item[1] for item in trainset]).to(device)
    eye = torch.eye(num_classes).to(device)

    g = torch.Generator(device=device)
    g.manual_seed(init_seed)

    if approach == "opt":
        ds_tr = torch.utils.data.TensorDataset(X, eye[targets], targets)
        if balanced:
            sampler = BalancedBatchSampler(group_onehot=eye[targets], batch_size=batch_size,
                                           drop_last=True, generator=g)
            trainloader = torch.utils.data.DataLoader(ds_tr, batch_sampler=sampler, generator=g)
        else:
            trainloader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)
        # cv_seed (not init_seed) -> same eval subsample across init seeds.
        opt_eval = _balanced_subsample(X, targets, eye, num_classes, opt_eval_size, cv_seed, device)
        return trainloader, None, None, opt_eval

    # K-fold the training set (stratified by class); held fold = validation.
    strat = targets.cpu().numpy()
    tr_rel, val_rel = list(StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=cv_seed
    ).split(np.arange(len(targets)), strat))[fold]
    tr_idx = torch.as_tensor(tr_rel, device=device)
    val_idx = torch.as_tensor(val_rel, device=device)

    Xtr, ttr = X[tr_idx], targets[tr_idx]
    Xval, tval = X[val_idx], targets[val_idx]
    ds_tr = torch.utils.data.TensorDataset(Xtr, eye[ttr], ttr)
    ds_val = torch.utils.data.TensorDataset(Xval, eye[tval], tval)

    if balanced:
        sampler = BalancedBatchSampler(group_onehot=eye[ttr], batch_size=batch_size,
                                       drop_last=True, generator=g)
        trainloader = torch.utils.data.DataLoader(ds_tr, batch_sampler=sampler, generator=g)
    else:
        trainloader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)
    valloader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, generator=g)

    # Fixed held-out test = canonical test split.
    testset = ds_cls(root='./data', train=False, download=True, transform=transform)
    X_test = torch.stack([item[0] for item in testset]).to(device)
    t_test = torch.tensor([item[1] for item in testset]).to(device)
    ds_test = torch.utils.data.TensorDataset(X_test, eye[t_test], t_test)
    testloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, generator=g)

    return trainloader, valloader, testloader, None
