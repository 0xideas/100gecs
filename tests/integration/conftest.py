import os
import shutil

import numpy as np
import pytest

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
    gec.set_gec_hyperparameters(
        {
            "l": 1.0,
            "l_bagging": 0.1,
            "hyperparams_acquisition_percentile": 0.7,
            "bagging_acquisition_percentile": 0.7,
            "bandit_greediness": 1.0,
            "n_random_exploration": 2,
            "n_sample": 100,
            "n_sample_initial": 100,
            "best_share": 0.2,
            "hyperparameters": [
                "learning_rate",
                "n_estimators",
                "num_leaves",
                "max_bin",
                "reg_alpha",
                "reg_lambda",
                "min_child_samples",
                "min_child_weight",
                "colsample_bytree",  # feature_fraction
            ],
            "randomize": True,
        }
    )
    gec.fit(X, y, 10)
    return gec


@pytest.fixture(scope="session")
def serialisation_path():
    return "tests/data/outputs/gec.json"


@pytest.fixture(scope="session")
def gec_is_serialised(gec, serialisation_path):
    gec.serialize(serialisation_path)
