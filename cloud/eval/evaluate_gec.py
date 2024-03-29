import json
import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import boto3
import numpy as np
import typer
from cloud_io.load_dataset import load_dataset
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

from gecs.lightgec import LightGEC

VERSION = 34
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
    dataset: str = "bank",
    static_seed: bool = False,
    dataset_path: Optional[str] = None,
):

    client = boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    X, y = load_dataset(dataset, dataset_path)

    # load hyperparameters to evaluate
    with open(config_path, "r") as f:
        hyperparameter_dicts = json.loads(f.read())

    if not static_seed:
        np.random.seed(int(datetime.now().timestamp() % 1 * 1e7))

    for hyperparameter_dict in hyperparameter_dicts:
        random_id = "".join(list(np.random.randint(0, 10, size=6).astype(str)))
        gec = LightGEC()
        n_iters = hyperparameter_dict.pop("n_iters")
        gec.set_gec_hyperparameters(hyperparameter_dict)
        hyperparameters = hyperparameter_dict.pop("hyperparameters")
        for i, n_iter in enumerate(n_iters):
            hyperparameter_representation = (
                "-".join([f"{k}{v}" for k, v in hyperparameter_dict.items()])
                + f"_niter{n_iter}"
                + f"_{random_id}"
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
                    "hyperparameters": "-".join(hyperparameters),
                    "n_iter": n_iter,
                    "cv_score": score_bayes,
                    "model_name": "-".join(
                        [f"{k}{v}" for k, v in hyperparameter_dict.items()]
                    )
                    + f"_{random_id}",
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

        figs = gec.plot_gec()
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
