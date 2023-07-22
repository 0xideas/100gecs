from cloud_io.load_bank_dataset import load_bank_dataset
from cloud_io.load_cover_dataset import load_cover_dataset
from cloud_io.load_income_dataset import load_income_dataset
from cloud_io.load_mushroom_dataset import load_mushroom_dataset
from cloud_io.load_enzymes_dataset import load_enzymes_dataset


default_paths = {
    "bank": "/home/ubuntu/data/bank/bank-full.csv",
    "income": "/home/ubuntu/data/income/income.csv",
    "cover": "/home/ubuntu/data/cover/cover.csv",
    "mushroom": "/home/ubuntu/data/mushroom/mushroom.csv",
    "enzymes1": "/home/leon/data/enzymes/enzymes-train.csv",
    "enzymes2": "/home/leon/data/enzymes/enzymes-train.csv"
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
    elif dataset == "enzymes1":
        X, y = load_enzymes_dataset(dataset_path, 1.0, target=1)
    elif dataset == "enzymes2":
        X, y = load_enzymes_dataset(dataset_path, 1.0, target=2)
    else:
        raise Exception(f"dataset '{dataset}' is not available")
    return X, y
