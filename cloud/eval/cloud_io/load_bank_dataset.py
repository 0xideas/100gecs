import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_bank_dataset(path, share=0.1):
    data = pd.read_csv(path, sep=";")
    categorical_columns = ["job", "marital", "education", "contact", "poutcome", "month"]
    binary = ["default", "housing", "loan", "y"]

    enc = OneHotEncoder()

    def yesNoBinary(column):
        return pd.DataFrame(
            [1 if value == "yes" else 0 for value in column], columns=[column.name]
        )

    one_hot = pd.DataFrame(
        enc.fit_transform(data[categorical_columns]).toarray(),
        columns=enc.get_feature_names_out(categorical_columns),
    )
    data2 = pd.concat(
        [one_hot]
        + [
            yesNoBinary(data[col]) if col in binary else data[col]
            for col in data.columns
        ],
        1,
    ).drop(categorical_columns, 1)
    X, y = data2.values[:, :-1], data2.values[:, -1]
    np.random.seed(102)
    ind = np.random.uniform(0, 1, X.shape[0]) < share
    X = X[ind, :]
    y = y[ind]

    X_pos = X[y == 1, :]
    y_pos = y[y == 1]

    X_reweighted = np.concatenate([X] + [X_pos] * 3, axis=0)
    y_reweighted = np.concatenate([y] + [y_pos] * 3)

    return (X_reweighted, y_reweighted)
