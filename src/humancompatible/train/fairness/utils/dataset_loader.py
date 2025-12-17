from fairml_datasets import Dataset
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data_dutch(test_size=0.2, seed_n = 42):
    """
    Loads the dutch dataset with the classification of predicting the income class.
    Sensitive features are [sex, age].
    """

    # Get the dataset
    dataset = Dataset.from_id("dutch") 

    # Load as pandas DataFrame
    df = dataset.load() 

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
    num_groups = 12 * 2

    # Map each combination to a unique index
    group_indices = sex_idx * 12 + (age-4)  # 0-24 groups

    # One-hot encode the combinations
    groups_onehot = np.eye(num_groups)[group_indices]

    # Create dictionary mapping index to combination
    group_dict = {}
    for i, (s, m) in enumerate(product(sex_cols, np.array(range(0, 12)) + 4)):
        group_dict[i] = f"{s} + age_{m}"

    # split
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        df_features, df_labels, groups_onehot, test_size=test_size, random_state=seed_n
    )

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, groups_train, groups_test


if __name__ == '__main__':

    get_data_dutch()