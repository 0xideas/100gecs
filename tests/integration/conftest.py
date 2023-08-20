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
def gec(X, y_class, gec_hps):
    gec = LightGEC()
    gec.set_gec_hyperparameters(gec_hps)
    gec.fit(X, y_class, 10)
    return gec


@pytest.fixture(scope="session")
def gecat(X, y_class, gec_hps):
    gecat = GECat()
    gecat.set_gec_hyperparameters(gec_hps)
    gecat.fit(X, y_class, 5)
    return gecat


@pytest.fixture(scope="session")
def ger(X, y_real, gec_hps):
    ger = GER()
    ger.set_gec_hyperparameters(gec_hps)
    ger.fit(X, y_real, 20)
    return ger


@pytest.fixture(scope="session")
def gecar(X, y_real, gec_hps):
    gecar = GECar()
    gecar.set_gec_hyperparameters(gec_hps)
    gecar.fit(X, y_real, 5)
    return gecar


@pytest.fixture(scope="session")
def gec_serialisation_path():
    return "tests/data/outputs/gec.json"


@pytest.fixture(scope="session")
def gec_is_serialised(gec, gec_serialisation_path):
    gec.serialize(gec_serialisation_path)


@pytest.fixture(scope="session")
def ger_serialisation_path():
    return "tests/data/outputs/ger.json"


@pytest.fixture(scope="session")
def ger_is_serialised(ger, ger_serialisation_path):
    ger.serialize(ger_serialisation_path)


@pytest.fixture(scope="session")
def gecat_serialisation_path():
    return "tests/data/outputs/gecat.json"


@pytest.fixture(scope="session")
def gecat_is_serialised(gecat, gecat_serialisation_path):
    gecat.serialize(gecat_serialisation_path)


@pytest.fixture(scope="session")
def gecar_serialisation_path():
    return "tests/data/outputs/gecar.json"


@pytest.fixture(scope="session")
def gecar_is_serialised(gecar, gecar_serialisation_path):
    gecar.serialize(gecar_serialisation_path)
