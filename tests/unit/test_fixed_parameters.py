import numpy as np
import pytest

from gecs.catgec import CatGEC
from gecs.catger import CatGER
from gecs.lightgec import LightGEC
from gecs.lightger import LightGER

lightgecs = [(LightGEC, "y_class"), (LightGER, "y_real")]
fixed_hyperparameters = [
    ["n_estimators", "num_leaves"],
    ["boosting_type", "colsample_bytree", "min_child_weight", "learning_rate"],
]
lightgecs_expanded = [
    (class_, switch_, fixed_hps)
    for class_, switch_ in lightgecs
    for fixed_hps in fixed_hyperparameters
]

# catgecs = [CatGEC, CatGER]


@pytest.mark.parametrize("gec_class,y_switch,fixed_hyperparameters", lightgecs_expanded)
def test_fixed_parameters_lightgecs(
    gec_class,
    y_switch,
    fixed_hyperparameters,
    lightgecs_params,
    X,
    y_class,
    y_real,
    return_monkeypatch_gecs_class,
):
    gec = gec_class(**lightgecs_params)

    gec = return_monkeypatch_gecs_class(gec)

    y = y_class if y_switch == "y_class" else y_real
    gec.fit(X, y, fixed_hyperparameters=fixed_hyperparameters)

    tried_hyperparameters = gec.tried_hyperparameters()

    for tried_hyperparameter_combination in tried_hyperparameters:
        for fixed_hyperparameter in fixed_hyperparameters:
            assert (
                tried_hyperparameter_combination[fixed_hyperparameter]
                == lightgecs_params[fixed_hyperparameter]
            )

    variable_hyperparameters = set(gec._optimization_candidate_init_args).difference(
        set(fixed_hyperparameters)
    )
    for variable_hyperparameter in variable_hyperparameters:
        hyperparameter_present_count = np.sum(
            [
                variable_hyperparameter in tried_hyperparameter_combination
                for tried_hyperparameter_combination in tried_hyperparameters
            ]
        )
        assert hyperparameter_present_count == len(tried_hyperparameters) or (
            variable_hyperparameter == "subsample"
            and (hyperparameter_present_count > 0)
        )
        hyperparameter_values = [
            tried_hyperparameter_combination[variable_hyperparameter]
            for tried_hyperparameter_combination in tried_hyperparameters
            if variable_hyperparameter in tried_hyperparameter_combination
        ]
        assert len(np.unique(hyperparameter_values)) > 1
