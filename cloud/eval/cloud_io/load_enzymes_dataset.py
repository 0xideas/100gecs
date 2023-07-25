import numpy as np
import pandas as pd


def load_enzymes_dataset(path, share, target):
    data = pd.read_csv(path, sep=",", header=0)

    y = data[f"EC{target}"].values
    X = data.iloc[:, 1:-6].values

    return (X, y)
