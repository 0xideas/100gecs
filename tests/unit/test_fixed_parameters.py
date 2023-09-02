import numpy as np
import pytest

from gecs.catgec import CatGEC
from gecs.catger import CatGER
from gecs.lightgec import LightGEC
from gecs.lightger import LightGER

lightgecs = [(LightGEC, "y_class", "light"), (LightGER, "y_real", "light")]
fixed_hyperparameters_lightgecs = [
    ["n_estimators", "num_leaves"],
    ["boosting_type", "colsample_bytree", "min_child_weight", "learning_rate"],
]
lightgecs_expanded = [
    (class_, y_switch, gec_switch, fixed_hps)
    for class_, y_switch, gec_switch in lightgecs
    for fixed_hps in fixed_hyperparameters_lightgecs
]

fixed_hyperparameters_catgecs = [
    ["n_estimators", "num_leaves"],
    ["bootstrap_type", "colsample_bylevel", "min_child_samples", "learning_rate"]
]

catgecs = [(CatGEC, "y_class", "catgec"), (CatGER, "y_real", "catger")]
catgecs_expanded = [
    (class_, y_switch, gec_switch, fixed_hps)
    for class_, y_switch, gec_switch in lightgecs
    for fixed_hps in fixed_hyperparameters_catgecs
]


@pytest.mark.parametrize("gec_class,y_switch,gec_switch,fixed_hyperparameters", (lightgecs_expanded))
def test_fixed_parameters_lightgecs(
    gec_class,
    y_switch,
    gec_switch,
    fixed_hyperparameters,
    lightgecs_params,
    catgec_params,
    catger_params,
    X,
    y_class,
    y_real,
    return_monkeypatch_gecs_class,
):
    if gec_switch == "light":
        params = lightgecs_params
    elif gec_switch == "catgec":
        params = catgec_params
    elif gec_switch == "catger":
        params = catger_params
    else:
        pass

    
    gec = gec_class(**params)

    gec = return_monkeypatch_gecs_class(gec)

    y = y_class if y_switch == "y_class" else y_real
    gec.fit(X, y, fixed_hyperparameters=fixed_hyperparameters)

    tried_hyperparameters = gec.tried_hyperparameters()
    variable_hyperparameters = set(gec._optimization_candidate_init_args).union(set([s[0] for s in gec.categorical_hyperparameters[:-1]])).difference(
        set(fixed_hyperparameters)
    )

    for tried_hyperparameter_combination in tried_hyperparameters:
        for fixed_hyperparameter in fixed_hyperparameters:
            assert (
                tried_hyperparameter_combination[fixed_hyperparameter]
                == lightgecs_params[fixed_hyperparameter]
            )
        
        for tried_param, tried_value in tried_hyperparameter_combination.items():
            if tried_param not in variable_hyperparameters.union({"subsample_freq"}):
                assert tried_value == lightgecs_params[tried_param], tried_param


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
