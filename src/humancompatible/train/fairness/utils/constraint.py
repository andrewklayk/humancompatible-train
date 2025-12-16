import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from folktables import ACSDataSource, ACSIncome, generate_categories
from itertools import product
from humancompatible.train.fairness.utils import BalancedBatchSampler
from torch.nn import Sequential
from humancompatible.train.optim.ssl_alm_adam import SSLALM_Adam
from fairret.statistic.linear_fractional import PositiveRate

def positive_rate(out_batch, prob_f=torch.nn.functional.sigmoid):
    """
    Calculates the equal opportunity based on the given outputs of the model for the given groups. 
    
    """

    # compute the probabilities - using sigmoid (since that is used )
    if prob_f is None: 
        preds = out_batch
    else: 
        preds = prob_f( out_batch )

    pr = PositiveRate()
    probs_per_group = pr(preds, batch_sens)  # P(y=1|Sk = 1)

    return probs_per_group




if __name__ == '__main__':

        # define the torch seed here
    seed_n = 1
    n_epochs = 20

    # set the same seed for fair comparisons
    torch.manual_seed(seed_n)

    # log path file
    log_path = "./data/logs/log_benchmark_stochastic_2.npz"

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
        "MAR_Separated",
        "MAR_Widowed",
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
        "MAR_Separated",
        "MAR_Widowed",
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
    torch.manual_seed(seed_n)

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        features, labels, groups_onehot, test_size=0.2, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # tensor the train data
    features_train = torch.tensor(X_train, dtype=torch.float32)
    labels_train = torch.tensor(y_train, dtype=torch.float32)
    sens_train = torch.tensor(groups_train)

    # tensor the test data
    features_test = torch.tensor(X_test, dtype=torch.float32)
    labels_test = torch.tensor(y_test, dtype=torch.float32)
    sens_test = torch.tensor(groups_test)

    # create a train dataset    
    dataset_train = torch.utils.data.TensorDataset(features_train, sens_train, labels_train)

    # create the balanced dataloader
    sampler = BalancedBatchSampler(
        group_onehot=sens_train, batch_size=10, drop_last=True
    )
    balanced_sampler = torch.utils.data.DataLoader(dataset_train, batch_sampler=sampler)
    

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
        lr=0.05,  # primal variable lr
        dual_lr=0.08,  # lr of a dual ALM variable
        dual_bound=5,
        rho=1,  # rho penalty in ALM parameter
        mu=2,  # smoothing parameter
    )


    for batch_input, batch_sens, batch_label in balanced_sampler:

        # print(batch_input)
        # print(batch_sens)
        # print(batch_label)

        out = model_con(batch_input)

        equal_opportunity(out, batch_sens)

        exit()
