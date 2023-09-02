import json
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytest
from numpy import float64, ndarray


@pytest.fixture(scope="session")
def seed():
    return 102


@pytest.fixture(scope="session")
def X(seed):
    np.random.seed(seed)
    return np.random.randn(300, 3)


@pytest.fixture(scope="session")
def y_real(X, seed):
    return X.sum(1) ** 2 + np.random.uniform(0, 1, X.shape[0])


@pytest.fixture(scope="session")
def y_class(y_real, seed):
    np.random.seed(seed)
    return np.array([min(4, yy.astype(int)) for yy in y_real])


@pytest.fixture(scope="session")
def lightgecs_params():
    return {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample_for_bin": 200000,
        "objective": None,
        "class_weight": None,
        "min_split_gain": 0.0,
        "min_child_weight": 1e-3,
        "min_child_samples": 20,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": None,
        "n_jobs": -1,
        "silent": "warn",
        "importance_type": "split",
        "frozen": False,
    }


def monkey_patch_gecs_class(class_):
    def single_peak_real_hps_fn(
        params: Dict[str, Optional[Union[str, float, int, float64]]]
    ):
        score_constituents = {}

        score_constituents["learning_rate_loss"] = abs(params["learning_rate"] - 0.5)

        score_constituents["reg_alpha_loss"] = abs(params["reg_alpha"] - 2.5)

        score_constituents["reg_lambda_loss"] = abs(params["reg_lambda"] - 0.2)

        score_constituents["min_child_weight_loss"] = (
            abs(params["min_child_weight"] - 0.1) * 10
        )

        score_constituents["min_child_samples_loss"] = abs(
            params["min_child_samples"] - 5
        )

        score_constituents["colsample_bytree_loss"] = abs(
            params["colsample_bytree"] - 0.7
        )

        score = (20.0 - np.sum(list(score_constituents.values()))) / 20.0
        return (score, [score_constituents])

    def _calculate_cv_score_monkeypatch(
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
        class_: Any,
    ) -> float64:
        score, data = single_peak_real_hps_fn(params)

        return score

    class_._calculate_cv_score = _calculate_cv_score_monkeypatch

    return class_


@pytest.fixture(scope="session")
def return_monkeypatch_gecs_class():
    return monkey_patch_gecs_class
