from numpy import ndarray
from typing import List
from .gec_base import GECBase

class GEC(GECBase):

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