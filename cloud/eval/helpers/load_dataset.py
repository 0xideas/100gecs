from helpers.load_bank_dataset import load_bank_dataset
from helpers.load_income_dataset import load_income_dataset
from helpers.load_cover_dataset import load_cover_dataset
from helpers.load_mushroom_dataset import load_mushroom_dataset

default_paths = {
    "bank": "/home/ubuntu/data/bank/bank-full.csv",
    "income": "/home/ubuntu/data/income/income.csv",
    "cover": "/home/ubuntu/data/cover/cover.csv",
    "mushroom": "/home/ubuntu/data/mushroom/mushroom.csv",
}


def load_dataset(dataset, dataset_path=None):
    # load data
    if dataset_path is None:
        dataset_path = default_paths[dataset]

    if dataset == "bank":
        X, y = load_bank_dataset(dataset_path, 0.2)
    elif dataset == "income":
        X, y = load_income_dataset(dataset_path, 1.0)
    elif dataset == "cover":
        X, y = load_cover_dataset(dataset_path, 0.05)
    elif dataset == "mushroom":
        X, y = load_mushroom_dataset(dataset_path, 1.0)
    else:
        raise Exception(f"dataset '{dataset}' is not available")
    return X, y
