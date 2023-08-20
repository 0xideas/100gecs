import os
import shutil

import numpy as np
import pytest

from gecs.lightgec import LightGEC
from gecs.catger import CatGER
from gecs.catgec import CatGEC
from gecs.lightger import LightGER


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
def y_real(X, seed):
    return X.sum(1) ** 2 + np.random.uniform(0, 1, X.shape[0])


@pytest.fixture(scope="session")
def y_class(y_real, seed):
    np.random.seed(seed)
    return np.array([min(4, yy.astype(int)) for yy in y_real])


@pytest.fixture(scope="session")
def gec_hps():
    return {
        "l": 1.0,
        "hyperparams_acquisition_percentile": 0.7,
        "bandit_greediness": 1.0,
        "score_evaluation_method": None,
        "maximize_score": True,
        "n_random_exploration": 2,
        "n_sample": 100,
        "n_sample_initial": 100,
        "best_share": 0.2,
        "distance_metric": "cityblock",
        "hyperparameters": [
            "learning_rate",
            "num_leaves",
            "max_bin",
            "reg_lambda",
            "min_child_samples",
        ],
        "randomize": True,
    }


@pytest.fixture(scope="session")
def lightgec(X, y_class, gec_hps):
    gec = LightGEC()
    gec.set_gec_hyperparameters(gec_hps)
    gec.fit(X, y_class, 10)
    return gec


@pytest.fixture(scope="session")
def lightger(X, y_real, gec_hps):
    ger = LightGER()
    ger.set_gec_hyperparameters(gec_hps)
    ger.fit(X, y_real, 20)
    return ger


@pytest.fixture(scope="session")
def catgec(X, y_class, gec_hps):
    catgec = CatGEC()
    catgec.set_gec_hyperparameters(gec_hps)
    catgec.fit(X, y_class, 5)
    return catgec

@pytest.fixture(scope="session")
def catger(X, y_real, gec_hps):
    catger = CatGER()
    catger.set_gec_hyperparameters(gec_hps)
    catger.fit(X, y_real, 5)
    return catger


@pytest.fixture(scope="session")
def lightgec_serialisation_path():
    return "tests/data/outputs/gec.json"


@pytest.fixture(scope="session")
def lightgec_is_serialised(lightgec, lightgec_serialisation_path):
    lightgec.serialize(lightgec_serialisation_path)


@pytest.fixture(scope="session")
def lightger_serialisation_path():
    return "tests/data/outputs/ger.json"


@pytest.fixture(scope="session")
def lightger_is_serialised(lightger, lightger_serialisation_path):
    lightger.serialize(lightger_serialisation_path)


@pytest.fixture(scope="session")
def catgec_serialisation_path():
    return "tests/data/outputs/catgec.json"


@pytest.fixture(scope="session")
def catgec_is_serialised(catgec, catgec_serialisation_path):
    catgec.serialize(catgec_serialisation_path)


@pytest.fixture(scope="session")
def catger_serialisation_path():
    return "tests/data/outputs/catger.json"


@pytest.fixture(scope="session")
def catger_is_serialised(catger, catger_serialisation_path):
    catger.serialize(catger_serialisation_path)
