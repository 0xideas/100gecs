import copy
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from lightgbm import LGBMClassifier
from lightgbm.basic import Booster, LightGBMError
from lightgbm.compat import SKLEARN_INSTALLED
from lightgbm.sklearn import (
    LGBMModel,
    _EvalResultDict,
    _LGBM_BoosterBestScoreType,
    _LGBM_CategoricalFeatureConfiguration,
    _LGBM_FeatureNameConfiguration,
    _LGBM_InitScoreType,
    _LGBM_LabelType,
    _LGBM_ScikitCustomObjectiveFunction,
    _LGBM_ScikitEvalMetricType,
    _LGBM_ScikitMatrixLike,
    _LGBM_ScikitValidSet,
    _LGBM_WeightType,
)
from numpy import float64

from .gec_base import GECBase


class LightGEC(LGBMClassifier, GECBase):
    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, _LGBM_ScikitCustomObjectiveFunction]] = None,
        class_weight: Optional[Union[Dict, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-3,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        importance_type: str = "split",
        frozen: bool = False,
        **kwargs,
    ) -> None:
        adapted_lgbm_params = (
            str(inspect.signature(LGBMClassifier.__init__))
            .replace(
                "importance_type: str = 'split'",
                "importance_type: str = 'split', frozen: bool = False",
            )
            .replace("**kwargs)", "**kwargs) -> None")
        )
        gec_params = str(inspect.signature(LightGEC.__init__))
        assert (
            adapted_lgbm_params == gec_params
        ), f"{gec_params = } \n not equal to \n {adapted_lgbm_params = }"
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : str, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (Hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequency of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int or None, optional (default=None)
            Number of parallel threads to use for training (can be changed at prediction time by
            passing it as an extra keyword argument).

            For better performance, it is recommended to set this to the number of physical cores
            in the CPU.

            Negative integers are interpreted as following joblib's formula (n_cpus + 1 + n_jobs), just like
            scikit-learn (so e.g. -1 means using all threads). A value of zero corresponds the default number of
            threads configured for OpenMP in the system. A value of ``None`` (the default) corresponds
            to using the number of physical cores in the system (its correct detection requires
            either the ``joblib`` or the ``psutil`` util libraries to be installed).

            .. versionchanged:: 4.0.0

        importance_type : str, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess``,
        ``objective(y_true, y_pred, weight) -> grad, hess``
        or ``objective(y_true, y_pred, weight, group) -> grad, hess``:

            y_true : numpy 1-D array of shape = [n_samples]
                The target values.
            y_pred : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            weight : numpy 1-D array of shape = [n_samples]
                The weight of samples. Weights should be non-negative.
            group : numpy 1-D array
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : numpy 1-D array of shape = [n_samples] or numpy 2-D array of shape = [n_samples, n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, y_pred is a numpy 2-D array of shape = [n_samples, n_classes],
        and grad and hess should be returned in the same format.
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError(
                "scikit-learn is required for lightgbm.sklearn. "
                "You must install scikit-learn and restart your session to use this module."
            )

        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.importance_type = importance_type
        self._Booster: Optional[Booster] = None
        self._evals_result: _EvalResultDict = {}
        self._best_score: _LGBM_BoosterBestScoreType = {}
        self._best_iteration: int = -1
        self._other_params: Dict[str, Any] = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight: Optional[Union[Dict, str]] = None
        self._class_map: Optional[Dict[int, int]] = None
        self._n_features: int = -1
        self._n_features_in: int = -1
        self._classes: Optional[np.ndarray] = None
        self._n_classes: int = -1
        self.set_params(**kwargs)
        non_optimized_init_args = [
            "max_depth",
            "subsample_for_bin",
            "objective",
            "class_weight",
            "min_split_gain",
            "random_state",
            "n_jobs",
            "importance_type",
        ]
        optimization_candidate_init_args = [
            "learning_rate",
            "n_estimators",
            "num_leaves",
            "reg_alpha",
            "reg_lambda",
            "min_child_samples",
            "min_child_weight",
            "colsample_bytree",  # feature_fraction
            "subsample",
        ]
        categorical_hyperparameters = [
            ("boosting_type", ["gbdt", "dart", "rf"]),
        ]
        self._gec_init(
            frozen,
            non_optimized_init_args,
            optimization_candidate_init_args,
            categorical_hyperparameters,
        )

    def fit(
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        n_iter: int = 50,
        fixed_hyperparameters: List[str] = ["n_estimators", "num_leaves"],
        sample_weight: Optional[_LGBM_WeightType] = None,
        init_score: Optional[_LGBM_InitScoreType] = None,
        eval_set: Optional[List[_LGBM_ScikitValidSet]] = None,
        eval_names: Optional[List[str]] = None,
        eval_sample_weight: Optional[List[_LGBM_WeightType]] = None,
        eval_class_weight: Optional[List[float]] = None,
        eval_init_score: Optional[List[_LGBM_InitScoreType]] = None,
        eval_metric: Optional[_LGBM_ScikitEvalMetricType] = None,
        feature_name: _LGBM_FeatureNameConfiguration = "auto",
        categorical_feature: _LGBM_CategoricalFeatureConfiguration = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[str, Path, Booster, LGBMModel]] = None,
    ) -> "LightGEC":
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

    def _fit_best_params(self, X: _LGBM_ScikitMatrixLike, y: _LGBM_LabelType) -> None:

        if hasattr(self, "best_params") and self.best_params_ is not None:
            for k, v in self.best_params_.items():
                setattr(self, k, v)
            setattr(self, "random_state", 101)

        super().fit(X, y, **self.gec_fit_params_)

    def score_single_iteration(
        self,
        X: _LGBM_ScikitMatrixLike,
        y: _LGBM_LabelType,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
    ) -> float64:
        return self._calculate_cv_score(X, y, params, LGBMClassifier)

    def retrieve_hyperparameter(
        self, hyperparameter: str
    ) -> Optional[Union[int, str, float]]:
        return getattr(self, hyperparameter)

    def _process_arguments(
        self, arguments: Dict[str, Optional[Union[int, float, str]]]
    ) -> Dict[str, Optional[Union[int, float, str]]]:
        args = copy.deepcopy(arguments)
        bagging = args["gec_bagging"] == "gec_bagging_yes"
        if bagging:
            assert "subsample" in args
            args["subsample_freq"] = 1
        else:
            del args["subsample"]
        del args["gec_bagging"]
        return args
