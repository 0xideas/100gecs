import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


def load_income_dataset(path, share=1.0):
    data = pd.read_csv(path, sep=",", header=None)
    data.columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income-group",
    ]
    data = data.drop(columns=["fnlwgt", "relationship"])
    real_variables = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    categorical_variables = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "race",
        "sex",
        "native-country",
    ]

    enc = OneHotEncoder()
    one_hot = pd.DataFrame(
        enc.fit_transform(data[categorical_variables]).toarray(),
        columns=enc.get_feature_names_out(categorical_variables),
    )

    X = pd.concat([data[real_variables], one_hot], axis=1).values
    y = np.array([int(v == " >50K") for v in data["income-group"]])

    return (X, y)
