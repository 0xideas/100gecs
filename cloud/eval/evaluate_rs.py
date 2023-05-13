import json
import os
import typer
import shutil
import contextlib
import boto3

from io import BytesIO
import numpy as np
from datetime import datetime
from lightgbm import LGBMClassifier
from helpers.load_dataset import load_bank_dataset
from sklearn.model_selection import cross_val_score
from gecs100.gec import GEC
from sklearn.model_selection import RandomizedSearchCV

VERSION = 2
SCORE_LOCATION = f"eval/scores/v={VERSION}"
BUCKET = "100gecs"


app = typer.Typer(name="run random search benchmark")


def fit_random_search(X, y, gec, n_iter):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):

        classifier = LGBMClassifier()
        hyperparams = dict(
            gec.categorical_hyperparameters[:1] + gec._real_hyperparameters
        )
        random_search = RandomizedSearchCV(classifier, hyperparams, n_iter=n_iter)
        random_search.fit(X, y)

    return random_search


@app.command()
def run(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str = "eu-central-1",
    config_path: str = "/home/ubuntu/config.json",
    data_location: str = "/home/ubuntu/data/bank/bank-full.csv",
    dataset: str = "bank",
):
    client = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    # load data
    if dataset == "bank":
        X, y = load_bank_dataset(data_location, 0.2)
    else:
        raise Exception(f"dataset {dataset} is not available")

    np.random.seed(int(datetime.now().timestamp() % 1 * 1e7))
    gec = GEC()

    random_id = "".join(list(np.random.randint(0, 10, size=6).astype(str)))
    n_iters = [0, 20, 30, 40, 50, 70, 100, 150, 200]

    best_score = None
    best_params = None
    for last_n_iter, n_iter in zip(n_iters[:-1], n_iters[1:]):

        random_search = fit_random_search(X, y, gec, n_iter - last_n_iter)
        print(f"{best_score = } - {random_search.best_score_ = }")
        if best_score is None or best_score < random_search.best_score_:
            best_score = random_search.best_score_
            best_params = random_search.best_params_

        clf_rs = LGBMClassifier(**best_params)
        score_rs = np.mean(cross_val_score(clf_rs, X, y, cv=5))
        rs_result_repr = json.dumps(
            {
                "model_type": "random-search",
                "dataset": dataset,
                **dict(
                    zip(
                        [
                            "l",
                            "l_bagging",
                            "gaussian_variance_weight",
                            "bandit_greediness",
                        ],
                        [-1, -1, -1, -1],
                    )
                ),
                "n_iter": n_iter,
                "cv_score": score_rs,
                "model_name": f"random-search-{random_id}",
            }
        )
        response = client.put_object(
            Bucket=BUCKET,
            Body=rs_result_repr,
            Key=f"{SCORE_LOCATION}/random-search-niter{n_iter}-{random_id}.json",
        )


if __name__ == "__main__":
    app()
