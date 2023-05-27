import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


def load_income_dataset(path, share=1.0):
    data = pd.read_csv(path, sep=",", header=None)

    data.columns = [
        "edible",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]

    categorical_variables = list(data.columns)[1:]

    enc = OneHotEncoder()
    X = pd.DataFrame(
        enc.fit_transform(data[categorical_variables]).toarray(),
        columns=enc.get_feature_names_out(categorical_variables),
    ).values

    y = np.array([int(v == "p") for v in data["edible"]])

    return (X, y)
