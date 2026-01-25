from fairml_datasets import Dataset
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

def get_data_dutch(seed_n = 42, drop_small_groups=True, print_stats=True):
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

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(df_features, df_labels, groups_onehot, 
                                                                                   test_size=0.2, random_state=seed_n)
    X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(X_train, y_train, groups_train, test_size=0.25, random_state=seed_n)
    
    # print the statistics
    if print_stats:
        print("Number of Samples per group: \n")
        for idx in group_dict:
            print(f"{group_dict[idx]}, : {(groups_onehot[:, idx] == 1).sum()}")

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val, groups_train, groups_test, groups_val, group_dict


if __name__ == '__main__':

    get_data_dutch()