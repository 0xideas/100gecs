import pytest
import os
import shutil
import numpy as np
from gecs.gec import GEC


@pytest.fixture(scope="session", autouse=True)
def output_folder():
    os.makedirs("tests/data/outputs")
    os.makedirs("tests/data/outputs/plots")

    yield None
    shutil.rmtree("tests/data/outputs")


@pytest.fixture(scope="session")
def seed():
    return 102


@pytest.fixture(scope="session")
def X(seed):
    np.random.seed(seed)
    return np.random.randn(100, 3)


@pytest.fixture(scope="session")
def y(X, seed):
    np.random.seed(seed)
    return np.array(
        [
            min(4, yy)
            for yy in ((X.sum(1) ** 2 + np.random.uniform(0, 1, X.shape[0]))).astype(
                int
            )
        ]
    )


@pytest.fixture(scope="session")
def gec(X, y):
    gec = GEC()
    gec.set_gec_hyperparameters({
        "l": 1.0,
        "l_bagging": 0.1,
        "hyperparams_acquisition_percentile": 0.7,
        "bagging_acquisition_percentile": 0.7,
        "bandit_greediness": 1.0,
        "n_random_exploration": 10,
        "n_sample": 100,
        "n_sample_initial": 100,
        "best_share": 0.2,
        "hyperparameters": [
            "learning_rate",
            "n_estimators",
            "num_leaves",
            "max_bin",
            "lambda_l1",
            "lambda_l2",
            "min_data_in_leaf",
            "feature_fraction",
        ],
        "randomize": True,
        "estimators_leaves": {
            "n_exploitation": 5,
            "share_exploitation": 0.1,
            "exploration_n_estimators": 10,
            "exploration_num_leaves": 10
        }
    })
    gec.fit(X, y, 2, 10, 10)
    return gec
