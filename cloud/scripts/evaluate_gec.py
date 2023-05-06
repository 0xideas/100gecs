import json
import os
import typer
import shutil
import contextlib

import numpy as np
from lightgbm import LGBMClassifier
from helpers.load_dataset import load_bank_dataset
from sklearn.model_selection import cross_val_score
from gecs100.gec import GEC
from sklearn.model_selection import RandomizedSearchCV


N_ITER = 70


def prepare_folder(output_location):

    if os.path.exists(output_location) and os.path.isdir(output_location):
        shutil.rmtree(output_location)

    if os.path.exists(f"{output_location}.zip"):
        os.remove(f"{output_location}.zip")

    os.makedirs(output_location)
    os.makedirs(f"{output_location}/figures")


def fit_random_search(X, y, gec):

    classifier = LGBMClassifier()
    hyperparams = dict(gec.categorical_hyperparameters[:1] + gec.real_hyperparameters)
    random_search = RandomizedSearchCV(classifier, hyperparams, n_iter=N_ITER)
    random_search.fit(X, y)

    return random_search


def benchmark_against_random_search(X_eval, y_eval, gec, random_search):

    with contextlib.redirect_stdout(None):

        knn_bayes = LGBMClassifier(**gec.best_params_)
        score_bayes = np.mean(cross_val_score(knn_bayes, X_eval, y_eval, cv=5))
        knn_gs = LGBMClassifier(**random_search.best_params_)
        score_gs = np.mean(cross_val_score(knn_gs, X_eval, y_eval, cv=5))
        knn_default = LGBMClassifier()
        score_default = np.mean(cross_val_score(knn_default, X_eval, y_eval, cv=5))

    return {
        "bayesian": score_bayes,
        "random search": score_gs,
        "default": score_default,
    }


app = typer.Typer(name="run GEC benchmarking")


@app.command()
def run(
    output_location: str = "/home/ubuntu/output",
    data_location: str = "/home/ubuntu/data/bank/bank-full.csv",
    dataset: str = "bank",
):
    prepare_folder(output_location)

    gec = GEC()

    if dataset == "bank":
        X, y = load_bank_dataset(data_location, 0.2)
    else:
        raise Exception(f"dataset {dataset} is not available")

    gec.fit(X, y, N_ITER)

    gec.serialise(f"{output_location}/gec.json")
    gec.save_figs(f"{output_location}/figures/fig")

    random_search = fit_random_search(X, y, gec)
    benchmark = benchmark_against_random_search(X, y, gec, random_search)

    with open(f"{output_location}/benchmark.json", "w") as f:
        f.write(json.dumps(benchmark))

    shutil.make_archive(output_location, "zip", "./")


if __name__ == "__main__":
    app()
