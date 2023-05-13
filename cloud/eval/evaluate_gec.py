import json
import os
import typer
import shutil
import contextlib
import boto3

from io import BytesIO
import numpy as np
from lightgbm import LGBMClassifier
from helpers.load_dataset import load_bank_dataset
from sklearn.model_selection import cross_val_score
from gecs100.gec import GEC
from sklearn.model_selection import RandomizedSearchCV

VERSION = 2
SCORE_LOCATION = f"eval/scores/v={VERSION}"
ARTEFACT_LOCATION = f"eval/artefacts/v={VERSION}"
BUCKET = "100gecs"


app = typer.Typer(name="run GEC benchmarking")


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

    # load hyperparameters to evaluate
    with open(config_path, "r") as f:
        hyperparameter_dicts = json.loads(f.read())

    for hyperparameter_dict in hyperparameter_dicts:
        gec = GEC()
        n_iters = hyperparameter_dict.pop("n_iters")
        gec.set_gec_hyperparameters(hyperparameter_dict)
        for i, n_iter in enumerate(n_iters):
            hyperparameter_representation = (
                "_".join(
                    [f"{k.replace('_', '')}{v}" for k, v in hyperparameter_dict.items()]
                )
                + f"_niter{n_iter}"
            )
            if i == 0:
                gec.fit(X, y, n_iter)
            else:
                gec.fit(X, y, n_iter - n_iters[i - 1])

            knn_bayes = LGBMClassifier(**gec.best_params_)
            score_bayes = np.mean(cross_val_score(knn_bayes, X, y, cv=5))

            result_repr = json.dumps(
                {
                    "model_type": "gec",
                    "dataset": dataset,
                    **hyperparameter_dict,
                    "n_iter": n_iter,
                    "cv_score": score_bayes,
                }
            )
            response = client.put_object(
                Bucket=BUCKET,
                Body=result_repr,
                Key=f"{SCORE_LOCATION}/{hyperparameter_representation}.json",
            )

        gec_repr = json.dumps(gec._get_representation())
        response = client.put_object(
            Bucket=BUCKET,
            Body=gec_repr,
            Key=f"{ARTEFACT_LOCATION}/{hyperparameter_representation}.json",
        )

        figs = gec.summarise_gp_datas()
        for fig_name, fig in figs.items():
            buffer = BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            response = client.put_object(
                Bucket=BUCKET,
                Body=buffer,
                Key=f"{ARTEFACT_LOCATION}/{hyperparameter_representation}/{fig_name}.png",
            )


if __name__ == "__main__":
    app()
