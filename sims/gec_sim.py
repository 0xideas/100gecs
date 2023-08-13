import numpy as np
import pandas as pd
from numpy import ndarray, float64
from typing import Dict, Optional, Union

from gecs.gec import GEC

class GECSim(GEC):
    def set_sim_fn(self, fn):
        self.sim_fn = fn

    def _calculate_cv_score(
        self,
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
    ) -> float64:
        return(self.sim_fn(params))
    

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

optimal_params = {"boosting_type":"dart", "reg_alpha":2.5, "reg_lambda":0.2, "min_child_weight":0.1, "min_child_samples":5, "subsample":0.6, "subsample_freq": 0.5, "learning_rate": 0.5}

    
def single_peak_fn(params:  Dict[str, Optional[Union[str, float, int, float64]]]):
    score = 20.0
    if params["boosting_type"] == "dart":
        score += 5.0
    
    score -= abs(params["learning_rate"]-0.5)

    score -= abs(params["reg_alpha"]-2.5)

    score -= abs(params["reg_lambda"]-0.2)

    score -= abs(params["min_child_weight"]-0.1)*10

    score -= abs(params["min_child_samples"]-5)
    
    score -= abs(params["subsample"] - 0.6)

    score -= abs(params["subsample_freq"] - 5)

    return(score/20.0)



if __name__ == "__main__":
    gecsim = GECSim()
    gecsim.set_sim_fn(single_peak_fn)
    n_iter=50
    gec_hyperparameters = {
        "l": 1.0,
        "l_bagging": 0.1,
        "hyperparams_acquisition_percentile": 0.3,
        "bagging_acquisition_percentile": 0.3,
        "bandit_greediness": 0.4,
        "n_random_exploration": 5,
        "n_sample": 1000,
        "n_sample_initial": 1000,
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

    gecsim.fit(X, y, n_iter=n_iter)
    scores = gecsim.hyperparameter_scores_["output"]
    rolling_average_score = np.array([np.mean(scores[i:i+10]) for i in range(len(scores)-10)])
    print(f"max score {np.max(scores)} at {np.argmax(scores)}")
    print("rolling average score:")
    print(rolling_average_score)
    print("selected arms counts:")
    print(pd.Series(gecsim.selected_arms_).value_counts().sort_index())

