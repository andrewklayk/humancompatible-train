from folktables import ACSDataSource, generate_categories, ACSIncome, BasicProblem
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from humancompatible.train.fairness.utils import BalancedBatchSampler
from itertools import product


import itertools

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



def load_data_norm(batch_size=64):

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

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups, test_size=0.2, random_state=42
    )

    # split
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.25, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # make into a pytorch dataset, remove the sensitive attribute
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train, dtype=torch.float32)

    # make into a pytorch dataset, remove the sensitive attribute
    features_val = torch.tensor(X_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.float32)
    sens_val = torch.tensor(groups_val, dtype=torch.float32)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test, dtype=torch.float32)

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)
    dataset_val = torch.utils.data.TensorDataset(features_val, sens_val, labels_val)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=64, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=True)

    return (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val)


def load_data_FT(batch_size, sens_attrs, states, group_size_threshold = 0, sens_groups = None):
    # load folktables data
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    ACSProblem = BasicProblem(
        features=ACSIncome.features,
        target=ACSIncome.target,
        target_transform=ACSIncome.target_transform,
        group=sens_attrs,
        group_transform = lambda x: pd.get_dummies(x, columns=sens_attrs),
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

    df_sens_onehot = comb_cat_dummies(df_sens) if sens_groups else df_sens

    features = df_feat.drop(columns=[col for col in df_feat.columns if col.startswith(tuple(sens_attrs))]).to_numpy()
    groups = df_sens_onehot.to_numpy(dtype='float')
    labels = df_labels.to_numpy()

    if sens_groups:
        group_names = []
        for group in sens_groups:
            group_def = []
            for key, val in group.items():
                group_def.append("_".join([str(key),str(val)]))
            group_str = "&".join(group_def)
            group_names.append(group_str)

        keep_mask = df_sens_onehot[group_names].any(axis=1)
        features = features[keep_mask]
        labels = labels[keep_mask]
        df_sens_onehot = df_sens_onehot[keep_mask].drop(columns=[col for col in df_sens_onehot if col not in group_names])
        groups = df_sens_onehot.to_numpy()
    
    torch.manual_seed(0)
    
    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups, test_size=0.2, random_state=42
    )

    # split
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.25, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # print the statistics
    for idx in range(groups.shape[1]):
        print(f"{df_sens_onehot.columns[idx]}, : {(groups[:, idx] == 1).sum()}")

    # make into a pytorch dataset, remove the sensitive attribute
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train, dtype=torch.float32)
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

    # make into a pytorch dataset, remove the sensitive attribute
    features_val = torch.tensor(X_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.float32)
    sens_val = torch.tensor(groups_val, dtype=torch.float32)
    dataset_val = torch.utils.data.TensorDataset(features_val, labels_val)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test, dtype=torch.float32)
    dataset_test = torch.utils.data.TensorDataset(features_test, labels_test)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)
    dataset_val = torch.utils.data.TensorDataset(features_val, sens_val, labels_val)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=batch_size, drop_last=True
    )
    sampler_val = BalancedBatchSampler(
        group_onehot=sens_val, batch_size=batch_size, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=batch_size, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    return (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val)



def load_data_FT_prod(batch_size):

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

    # split
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.25, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
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
    features_val = torch.tensor(X_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.float32)
    sens_val = torch.tensor(groups_val)
    dataset_val = torch.utils.data.TensorDataset(features_val, labels_val)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test)
    dataset_test = torch.utils.data.TensorDataset(features_test, labels_test)

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)
    dataset_val = torch.utils.data.TensorDataset(features_val, sens_val, labels_val)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=batch_size, drop_last=True
    )
    sampler_val = BalancedBatchSampler(
        group_onehot=sens_val, batch_size=batch_size, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=batch_size, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    return (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val)



def load_data_FT_vec(batch_size, attr = "SEX"):

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

    if attr == 'SEX':
        sens_cols = ["SEX_Female", "SEX_Male"]
    elif attr == 'MAR':
        sens_cols = ['MAR_Married', 'MAR_Widowed', 'MAR_Divorced', "MAR_Separated", "MAR_Never married or under 15 years old"]
    features = df_feat.drop(columns=sens_cols).to_numpy(dtype="float")
    groups = df_feat[sens_cols].to_numpy(dtype="float")
    labels = df_labels.to_numpy(dtype="float")

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups, test_size=0.2, random_state=42
    )

    # split
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.25, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # make into a pytorch dataset, remove the sensitive attribute
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train, dtype=torch.float32)
    dataset_train = torch.utils.data.TensorDataset(features_train, labels_train)

    # make into a pytorch dataset, remove the sensitive attribute
    features_val = torch.tensor(X_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.float32)
    sens_val = torch.tensor(groups_val, dtype=torch.float32)
    dataset_val = torch.utils.data.TensorDataset(features_val, labels_val)

    # make into a pytorch dataset, remove the sensitive attribute
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test, dtype=torch.float32)
    dataset_test = torch.utils.data.TensorDataset(features_test, labels_test)

    # set the same seed for fair comparisons
    torch.manual_seed(0)

    # get the dataset
    dataset = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)
    dataset_val = torch.utils.data.TensorDataset(features_val, sens_val, labels_val)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # create a balanced sampling - needed for an unbiased gradient
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=batch_size, drop_last=True
    )
    sampler_val = BalancedBatchSampler(
        group_onehot=sens_val, batch_size=batch_size, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=batch_size, drop_last=True
    )

    # create a dataloader from the sampler
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test)

    return (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val)
    

def load_data_DUTCH(batch_size):
    # Get the data with a validation split
    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, group_names_dict = get_data_dutch(
        test_size=0.4, seed_n=42, drop_small_groups=True, print_stats=True
    )

    # Convert training data to PyTorch tensors
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape((-1, 1))
    sens_train = torch.tensor(groups_train)
    dataset_train = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # Convert validation data to PyTorch tensors
    features_val = torch.tensor(X_val, dtype=torch.float32)
    labels_val = torch.tensor(y_val, dtype=torch.float32).reshape((-1, 1))
    sens_val = torch.tensor(groups_val)
    dataset_val = torch.utils.data.TensorDataset(features_val, sens_val, labels_val)

    # Convert test data to PyTorch tensors
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape((-1, 1))
    sens_test = torch.tensor(groups_test)
    dataset_test = torch.utils.data.TensorDataset(features_test, sens_test, labels_test)

    # Create balanced samplers
    sampler_train = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=batch_size, drop_last=True
    )
    sampler_val = BalancedBatchSampler(
        group_onehot=sens_val, batch_size=252*4, drop_last=True
    )
    sampler_test = BalancedBatchSampler(
        group_onehot=sens_test, batch_size=252*4, drop_last=True
    )

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler_train, num_workers=8)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=8)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=8)

    return (dataloader_train, dataloader_val, dataloader_test), (features_train, sens_train, labels_train), (features_val, sens_val, labels_val)




from fairml_datasets import Dataset
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data_dutch(test_size=0.2, seed_n = 42, drop_small_groups=True, print_stats=True):
    """
    Loads the dutch dataset with the classification of predicting the income class.
    Sensitive features are [sex, age].

    drop_small_groups - if True, sensitive groups with few samples are dropped from the dataset
    """

    # Get the dataset
    dataset = Dataset.from_id("dutch") 

    # Load as pandas DataFrame
    df = dataset.load() 

    # drop high aged population
    if drop_small_groups:
        df = df.drop(df[(df.age == '13') | (df.age == '14') | (df.age == '15')].index)
        num_age_groups = 9
    else: 
        num_age_groups = 12 

    # Get the target column
    target_column = dataset.get_target_column()

    # Transform to e.g. impute missing data
    df_transformed, transformation_info = dataset.transform(df)
    
    # Sensitive columns may change due to transformation
    sensitive_columns = ['age', 'sex_1', 'sex_2']

    # get the labels
    df_labels = df_transformed[target_column] 

    # get the features - no labels and sensitive features
    df_features = df_transformed.drop(columns=target_column)
    df_features = df_features.drop(columns=sensitive_columns)

    sex_cols = ['sex_1', 'sex_2']

    # Convert each row to sex index and marital index
    sex_idx = df_transformed[sex_cols].values.argmax(axis=1).astype(int)
    age = df_transformed['age'].values.astype(int)

    # num groups 
    num_groups = num_age_groups * 2

    # Map each combination to a unique index
    group_indices = sex_idx * num_age_groups + (age-4)  # 0-24 groups

    # One-hot encode the combinations
    groups_onehot = np.eye(num_groups)[group_indices]

    # Create dictionary mapping index to combination
    group_dict = {}
    for i, (s, m) in enumerate(product(sex_cols, np.array(range(0, num_age_groups)) + 4)):
        group_dict[i] = f"{s} + age_{m}"

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        df_features, df_labels, groups_onehot, test_size=test_size, random_state=seed_n
    )

    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
        X_train, y_train, groups_train, test_size=0.25, random_state=seed_n
    )

    y_val = y_val.to_numpy()
    # print the statistics
    if print_stats:
        print("Number of Samples per group: \n")
        for idx in group_dict:
            print(f"{group_dict[idx]}, : {(groups_onehot[:, idx] == 1).sum()}")

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, group_dict

