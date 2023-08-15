import numpy as np
import pandas as pd
import json
from numpy import ndarray, float64
from typing import Dict, Optional, Union

from gecs.gec import GEC

class GECSim(GEC):
    def set_sim_fn(self, fn):
        self.sim_fn = fn
        self.sim_fn_data = []

    def _calculate_cv_score(
        self,
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
    ) -> float64:
        score, data = self.sim_fn(params)
        self.sim_fn_data.extend(data)
        return(score)
    

example_params =  {
  'boosting_type': 'dart',
  'learning_rate': 0.5476000000000003,
  'reg_alpha': 22.090000000000003,
  'reg_lambda': 0.07840000000000001,
  'min_child_weight': 8.100000000000002e-05,
  'min_child_samples': 6,
  'colsample_bytree': 0.13,
  'num_leaves': 31,
  'n_estimators': 100,
  'max_depth': -1,
  'subsample_for_bin': 200000,
  'objective': None,
  'class_weight': None,
  'min_split_gain': 0.0,
  'subsample': 0.7100000000000002,
  'subsample_freq': 2,
  'random_state': None,
  'n_jobs': -1,
  'silent': 'warn',
  'importance_type': 'split'
}

optimal_params = {"boosting_type":"dart", "reg_alpha":2.5, "reg_lambda":0.2, "min_child_weight":0.1, "min_child_samples":5, "subsample":0.6, "subsample_freq": 0.5, "learning_rate": 0.5, "colsample_bytree":0.7}

    
def single_peak_real_hps_fn(params:  Dict[str, Optional[Union[str, float, int, float64]]]):
    score_constituents = {}

    score_constituents["learning_rate_loss"] = abs(params["learning_rate"]-0.5)

    score_constituents["reg_alpha_loss"] = abs(params["reg_alpha"]-2.5)

    score_constituents["reg_lambda_loss"] = abs(params["reg_lambda"]-0.2)

    score_constituents["min_child_weight_loss"] = abs(params["min_child_weight"]-0.1)*10

    score_constituents["min_child_samples_loss"] = abs(params["min_child_samples"]-5)

    score_constituents["colsample_bytree_loss"] = abs(params["colsample_bytree"]-0.7)

    score = (20.0 - np.sum(list(score_constituents.values())))/20.0
    return(score, [score_constituents])


def calculate_rs_scores_real_hps(fn, gecsim, n_iter):
    hyperparams = dict(gecsim._real_hyperparameters)
    params = [{k:np.random.choice(v) for k,v in hyperparams.items()} for _ in range(n_iter)]
    scores = [fn(params_)[0] for params_ in params]
    return(scores)

if __name__ == "__main__":
    gecsim = GECSim()
    gecsim.set_sim_fn(single_peak_real_hps_fn)
    n_iter=50
    gec_hyperparameters = {
        "l": 1.0,
        "l_bagging": 0.1,
        "hyperparams_acquisition_percentile": 0.2,
        "bagging_acquisition_percentile": 0.2,
        "bandit_greediness": 0.4,
        "n_random_exploration": 10,
        "n_sample": 1000,
        "n_sample_initial": 2000,
        "best_share": 0.2,
        "hyperparameters": [
            "learning_rate",
            "n_estimators",
            "num_leaves",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "min_child_weight",
            "colsample_bytree",  # feature_fraction
        ],
        "randomize": True,
    }
    gecsim.set_gec_hyperparameters(gec_hyperparameters)

    np.random.seed(101)
    X = np.random.randn(100, 3)
    y = np.random.choice([0, 1], 100)

    #"reg_alpha", "colsample_bytree", "min_child_weight", "min_child_samples",
    gecsim.fit(X, y, n_iter=n_iter, fixed_hyperparameters=[ "num_leaves", "n_estimators"])
    scores = gecsim.hyperparameter_scores_["output"]
    rolling_average_score = np.array([np.mean(scores[i:i+10]) for i in range(len(scores)-10)])
    print(f"max score {np.max(scores)} at {np.argmax(scores)}")
    rs_scores = calculate_rs_scores_real_hps(single_peak_real_hps_fn, gecsim, n_iter)
    print(f"rs max score {np.max(rs_scores)} at {np.argmax(rs_scores)}")

    tried_hyperparameters = pd.DataFrame(gecsim.tried_hyperparameters())
    tried_hyperparameters["score"] = scores

    with open("sims/data/tried_hyperparameters.csv", "w") as f:
        tried_hyperparameters.to_csv(f, index=False)

    with open("sims/data/score_constituents.csv", "w") as f:
        pd.DataFrame(gecsim.sim_fn_data[:-2]).to_csv(f, index=False)