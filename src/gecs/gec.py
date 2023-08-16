import copy
from numpy import ndarray, float64
from typing import List, Dict, Optional, Union
from lightgbm import LGBMClassifier
from .gec_base import GECBase

class GEC(LGBMClassifier, GECBase):
    def __post_init__(self):
        self._gec_init()

    def fit(
        self,
        X: ndarray,
        y: ndarray,
        n_iter: int = 50,
        fixed_hyperparameters: List[str] = ["n_estimators", "num_leaves"],
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ) -> "GEC":
        """Docstring is inherited from the LGBMClassifier.

        Except for

        Parameters:
        ----------
            n_iter : int
                number of optimization steps
            fixed_hyperparameters : list[str]
                list of hyperparameters that should not be optimised
        """

        self.gec_fit_params_ = {
            "sample_weight": sample_weight,
            "init_score": init_score,
            "eval_set": eval_set,
            "eval_names": eval_names,
            "eval_sample_weight": eval_sample_weight,
            "eval_class_weight": eval_class_weight,
            "eval_init_score": eval_init_score,
            "eval_metric": eval_metric,
            "feature_name": feature_name,
            "categorical_feature": categorical_feature,
            "callbacks": callbacks,
            "init_model": init_model,
        }
        self._fit_inner(X, y, n_iter, fixed_hyperparameters)


    def __sklearn_clone__(self):
        gec = GEC()

        for k, v in self.__dict__.items():
            gec.__dict__[k] = copy.deepcopy(v)

        return gec
    
    def set_params(self, **kwargs) -> None:
        if "frozen" in kwargs:
            setattr(self, "frozen", kwargs.pop("frozen"))
        super().set_params(**kwargs)

    def get_params(
        self, deep: bool = True
    ) -> Dict[str, Optional[Union[str, float, int, bool]]]:
        if hasattr(self, "best_params_") and self.best_params_ is not None:
            params = copy.deepcopy(self.best_params_)
        else:
            params = super().get_params(deep)
        params["frozen"] = self.frozen

        return params
    
    def _fit_best_params(self, X: ndarray, y: ndarray) -> None:

        if hasattr(self, "best_params") and self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self, k, v)
            setattr(self, "random_state", 101)

        super().fit(X, y, **self.gec_fit_params_)

    def score_single_iteration(
        self,
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]]
    ):
        return(self._calculate_cv_score(X, y, params, LGBMClassifier))