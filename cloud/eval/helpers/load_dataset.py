from helpers.load_bank_dataset import load_bank_dataset
from helpers.load_income_dataset import load_income_dataset
from helpers.load_cover_dataset import load_cover_dataset


def load_dataset(dataset):
    # load data
    if dataset == "bank":
        X, y = load_bank_dataset("/home/ubuntu/data/bank/bank-full.csv", 0.2)
    elif dataset == "income":
        X, y = load_income_dataset("/home/ubuntu/data/income/income.csv", 1.0)
    elif dataset == "cover":
        X, y = load_cover_dataset("/home/ubuntu/data/cover/cover.csv", 1.0)
    else:
        raise Exception(f"dataset '{dataset}' is not available")
    return X, y
