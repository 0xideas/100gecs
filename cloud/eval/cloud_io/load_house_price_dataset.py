import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_house_price_dataset(path):
    data = pd.read_csv(path, sep=",")


    categorical_columns = [c for c in data.columns if data[c].dtype == "object"]
    number_columns = [c for c in list(data.columns)[1:-1] if c not in categorical_columns]


    data[categorical_columns] = data[categorical_columns].astype(str)

    enc = OneHotEncoder(handle_unknown="ignore")

    one_hot = pd.DataFrame(
        enc.fit_transform(data[categorical_columns]).toarray().astype(int),
        columns=enc.get_feature_names_out(categorical_columns),
    )

    X = pd.concat([data[number_columns], one_hot], axis=1).values
    y = data["SalePrice"].values

    return(X, y)
