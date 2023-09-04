import contextlib
import copy
import inspect
import itertools
import json
import math
import os
import warnings
from datetime import datetime
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from numpy import float16, float64, ndarray, str_
from numpy.random.mtrand import RandomState
from scipy.spatial.distance import cdist
from scipy.stats import beta
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import cartesian
from tqdm import tqdm


class GECBase:
    def _gec_init(
        self,
        frozen: bool,
        non_optimized_init_args: List[str],
        optimization_candidate_init_args: List[str],
        categorical_hyperparameters: List[Tuple[str, List[str]]],
    ) -> None:

        self.fix_boosting_type_ = False
        self.fix_bootstrap_type_ = False

        self.frozen = frozen

        self._init_args = {
            arg: self.retrieve_hyperparameter(arg) for arg in non_optimized_init_args
        }
        self._optimization_candidate_init_args = optimization_candidate_init_args

        self.gec_hyperparameters_ = {
            "l": 1.0,
            "hyperparams_acquisition_percentile": 0.7,
            "bandit_greediness": 1.0,
            "score_evaluation_method": None,
            "maximize_score": True,
            "n_random_exploration": 5,
            "n_sample": 1000,
            "n_sample_initial": 1000,
            "best_share": 0.2,
            "distance_metric": "cityblock",
            "hyperparameters": optimization_candidate_init_args,
            "randomize": True,
        }

        self.categorical_hyperparameters = categorical_hyperparameters + [
            ("gec_bagging", ["gec_bagging_yes", "gec_bagging_no"])
        ]
        self._categorical_hyperparameter_names = [
            cat[0] for cat in self.categorical_hyperparameters
        ]

        self._set_gec_attributes()
        self._set_gec_fields()

    def _set_gec_attributes(self) -> None:

        prohibited_combinations = [
            "rf-gec_bagging_no",
            "Plain-Bayesian-gec_bagging_yes",
            "Plain-No-gec_bagging_yes",
        ]

        self._categorical_hyperparameter_combinations = [
            "-".join(y)
            for y in itertools.product(
                *[x[1] for x in self.categorical_hyperparameters]
            )
            if "-".join(y) not in prohibited_combinations
        ]

        if self.fix_boosting_type_:
            self._categorical_hyperparameter_combinations = [
                hp_comb
                for hp_comb in self._categorical_hyperparameter_combinations
                if hp_comb.startswith(self.retrieve_hyperparameter("boosting_type"))
            ]

        if self.fix_bootstrap_type_:
            self._categorical_hyperparameter_combinations = [
                hp_comb
                for hp_comb in self._categorical_hyperparameter_combinations
                if hp_comb.split("-")[1]
                == self.retrieve_hyperparameter("bootstrap_type")
            ]

        real_hyperparameters_all_across_classes = [
            ("learning_rate", np.exp(np.arange(-6.5, 0, 0.2))),
            ("num_leaves", np.exp(np.arange(2, 7.5, 0.4)).astype(int)),
            (
                "n_estimators",
                np.unique((np.exp(np.arange(1, 7.5, 0.5)) * 3).astype(int)),
            ),
            ("reg_alpha", np.exp(np.arange(-20, 4, 0.5))),
            ("reg_lambda", np.exp(np.arange(-20, 4, 0.5))),
            ("min_child_weight", np.exp(np.arange(-20, 0.0, 1.0))),
            ("min_child_samples", np.arange(2, 50, 2)),
            ("colsample_bytree", np.arange(0.1, 1.00, 0.05)),
            ("colsample_bylevel", np.arange(0.1, 1.00, 0.05)),
            ("subsample", [x / 100 for x in range(10, 101, 5)]),
        ]
        self._real_hyperparameters_all = [
            (n, r)
            for n, r in real_hyperparameters_all_across_classes
            if n in self._optimization_candidate_init_args
        ]

        self.fixed_params = {
            hyperparameter: self.retrieve_hyperparameter(hyperparameter)
            for hyperparameter, _ in self._real_hyperparameters_all
            if hyperparameter not in self.gec_hyperparameters_["hyperparameters"]
        }

        self._real_hyperparameters = [
            (hp_name, range_)
            for hp_name, range_ in self._real_hyperparameters_all
            if hp_name in self.gec_hyperparameters_["hyperparameters"]
        ]
        self._real_hyperparameters_linear = [
            (name, np.arange(-1.0, 1.0, 2 / len(values)).round(4).astype(np.float16))
            for name, values in self._real_hyperparameters
        ]

        self._real_hyperparameters_map = {
            name: dict(zip(linear_values, real_values))
            for ((name, linear_values), (_, real_values)) in zip(
                self._real_hyperparameters_linear, self._real_hyperparameters
            )
        }

        self._real_hyperparameters_map_reverse = {
            name: dict(zip(real_values, linear_values))
            for ((name, linear_values), (_, real_values)) in zip(
                self._real_hyperparameters_linear, self._real_hyperparameters
            )
        }

        self._validate_parameter_maps()

        self._real_hyperparameter_names, self._real_hyperparameter_ranges = zip(
            *self._real_hyperparameters_linear
        )

        self._real_hypermarameter_types = [
            np.array(s).dtype for _, s in self._real_hyperparameters
        ]

    def _set_gec_fields(self) -> None:
        self.kernel = RBF(self.gec_hyperparameters_["l"])
        self.hyperparameter_scores_ = {
            "inputs": [],
            "output": [],
            "means": [],
            "sigmas": [],
        }

        self.best_score_gec_ = None
        self.best_params_ = None
        self.gec_fit_params_ = None
        self.n_iterations = 0

        # parameters for bandit
        self.rewards_ = {
            c: {"a": 1, "b": 1} for c in self._categorical_hyperparameter_combinations
        }
        self.selected_arms_ = []

        self.best_scores_gec_ = {}
        self.best_params_gec_ = {}

    def tried_hyperparameters(self):
        assert np.array(self.hyperparameter_scores_["inputs"]).shape[0] == len(
            self.selected_arms_
        )

        hyperparamters = []
        for selected_arm, hyperparameter_inputs in zip(
            self.selected_arms_, self.hyperparameter_scores_["inputs"]
        ):
            args = self._process_arguments(
                self._build_arguments(selected_arm, hyperparameter_inputs)
            )
            hyperparamters.append(args)

        return hyperparamters

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        self._kernel = value
        self.gaussian = GaussianProcessRegressor(kernel=value, n_restarts_optimizer=9)

    @property
    def gec_iter_(self) -> int:
        return len(self.hyperparameter_scores_["output"])

    @classmethod
    def _cast_to_type(cls, value, type_):
        if type_ == np.float64:
            return float(value)
        elif type_ == np.int64:
            return int(value)
        elif value is None:
            return None
        else:
            raise Exception(f"type {type_} currently not supported")

    @classmethod
    def deserialize(
        cls, path: str, X: Optional[ndarray] = None, y: Optional[ndarray] = None
    ):
        """Deserialize a model and fit underlying LGBMClassifier if X and y are provided

        Parameters
        ----------
            path : str
                path to serialized GEC
            X : np.ndarray, optional
                Input feature matrix
            y : np.ndarray, optional
                Target class labels

        Returns
        -------
            gec : GEC
                deserialized model object
        """
        with open(path, "r") as f:
            representation = json.loads(f.read())

        gec = cls()
        gec.gec_hyperparameters_ = representation["gec_hyperparameters"]
        gec.rewards_ = representation["rewards"]
        gec.selected_arms_ = representation["selected_arms"]
        gec.hyperparameter_scores_ = (
            gec._convert_gaussian_process_data_from_deserialisation(
                representation["hyperparameter_scores"]
            )
        )
        gec.best_params_ = representation["best_params"]
        gec.best_score_gec_ = float(representation["best_score_gec"])
        gec.best_params_gec_ = representation["best_params_gec"]
        gec.best_scores_gec_ = representation["best_scores_gec"]
        gec.gec_fit_params_ = representation["gec_fit_params"]

        if X is not None and y is not None:
            gec._fit_best_params(X, y)
        else:
            warnings.warn(
                "If X and y are not provided, the GEC model is not fitted for inference"
            )
        return gec

    @classmethod
    def _convert_gaussian_process_data_from_deserialisation(
        cls, data_dict: Dict[str, List[Union[List[float], float]]]
    ) -> Dict[str, List[Union[List[float], float]]]:
        converted_dict = {k: list(v) for k, v in data_dict.items()}
        return converted_dict

    def serialize(self, path: str) -> None:
        """Serialize GEC model object

        Parameters
        ----------
            path : str
                path to serialize GEC to
        """
        representation = self._get_representation()

        with open(path, "w") as f:
            f.write(json.dumps(representation))

    @classmethod
    def _convert_gaussian_process_data_for_serialisation(
        cls,
        data_dict: Dict[
            str,
            Union[
                List[Union[List[float], float64, int, ndarray]],
                List[Union[List[Union[float, float64]], float64, int, ndarray]],
            ],
        ],
    ) -> Dict[
        str,
        Union[
            List[Union[List[Union[float, float64]], float64, int, List[float64]]],
            List[Union[List[float], float64, int, List[float64]]],
        ],
    ]:
        def process_value(key, value):
            if not isinstance(value, np.ndarray):
                return value
            elif key != "inputs":
                return list(value)
            else:
                return list(value.astype(np.float64))

        converted_dict = {
            k2: [process_value(k2, vv) for vv in v] for k2, v in data_dict.items()
        }

        return converted_dict

    def _get_representation(self) -> Dict[str, Any]:
        hyperparameter_scores = self._convert_gaussian_process_data_for_serialisation(
            self.hyperparameter_scores_
        )
        representation = {
            "gec_hyperparameters": self.gec_hyperparameters_,
            "rewards": self.rewards_,
            "selected_arms": self.selected_arms_,
            "hyperparameter_scores": hyperparameter_scores,
            "best_params": self.best_params_,
            "best_score_gec": self.best_score_gec_,
            "best_params_gec": self.best_params_gec_,
            "best_scores_gec": self.best_scores_gec_,
            "gec_iter": self.gec_iter_,
            "gec_fit_params": {
                k: v
                for k, v in self.gec_fit_params_.items()
                if k not in ["log_cout", "log_cerr"]
            },
        }
        return representation

    def _validate_parameter_maps(self) -> None:
        real_to_linear_to_real = {
            v: self._real_hyperparameters_map[hp][
                self._real_hyperparameters_map_reverse[hp][v]
            ]
            == v
            for hp, values in self._real_hyperparameters
            for v in values
        }

        assert np.all(np.array(real_to_linear_to_real.values())), {
            k: v for k, v in real_to_linear_to_real.items() if not v
        }

        linear_to_real_to_linear = {
            v: self._real_hyperparameters_map_reverse[hp][
                self._real_hyperparameters_map[hp][v]
            ]
            == v
            for hp, values in self._real_hyperparameters_linear
            for v in values
        }
        assert np.all(np.array(linear_to_real_to_linear.values())), {
            k: v for k, v in linear_to_real_to_linear.items() if not v
        }

    def set_gec_hyperparameters(
        self, gec_hyperparameters: Dict[str, Union[int, float, List[str]]]
    ) -> None:
        """Set the hyperparameters of the GEC optimisation process

        Parameters
        ----------
            gec_hyperparameters : dict[str, float]
                dictionary with keys that are in self.gec_hyperparameters_
        """
        assert np.all(
            np.array(
                [hp in self.gec_hyperparameters_ for hp in gec_hyperparameters.keys()]
            )
        )
        self.gec_hyperparameters_.update(gec_hyperparameters)
        self._set_gec_attributes()

    def freeze(self):
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        return self

    def set_params(self, **kwargs) -> None:
        if "frozen" in kwargs:
            self["frozen"] = kwargs.pop("frozen")
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

    def _fit_inner(
        self, X: ndarray, y: ndarray, n_iter: int, fixed_hyperparameters: List[str]
    ):

        self.fix_boosting_type_ = "boosting_type" in fixed_hyperparameters
        self.fix_bootstrap_type_ = "bootstrap_type" in fixed_hyperparameters
        fixed_hyperparameters = [
            hp
            for hp in fixed_hyperparameters
            if hp not in ["boosting_type", "bootstrap_type"]
        ]

        if not self.frozen:
            filtered_hyperparameters = list(
                set(self.gec_hyperparameters_["hyperparameters"]).difference(
                    set(fixed_hyperparameters)
                )
            )
            self.set_gec_hyperparameters({"hyperparameters": filtered_hyperparameters})

            (
                self.best_params_gec_["search"],
                self.best_scores_gec_["search"],
            ) = self._optimise_hyperparameters(n_iter, X, y)

            best_params_grid = self._find_best_parameters()
            self.best_params_gec_["grid"] = self._process_arguments(
                self._replace_fixed_args(best_params_grid)
            )
            self.best_scores_gec_["grid"] = self.score_single_iteration(
                X, y, self.best_params_gec_["grid"]
            )

            best_params_grid_from_search = self._find_best_parameters_from_search(
                self.best_arm_, self.best_params_raw_
            )
            self.best_params_gec_["grid_from_search"] = self._process_arguments(
                self._replace_fixed_args(best_params_grid_from_search)
            )
            self.best_scores_gec_["grid_from_search"] = self.score_single_iteration(
                X, y, self.best_params_gec_["grid_from_search"]
            )

            for source, score in self.best_scores_gec_.items():
                if self.best_score_gec_ is None or score > self.best_score_gec_:
                    self.best_score_gec_ = score
                    self.best_params_ = self.best_params_gec_[source]

        self._fit_best_params(X, y)

        return self

    def _calculate_cv_score(
        self,
        X: ndarray,
        y: ndarray,
        params: Dict[str, Optional[Union[str, float, int, float64]]],
        class_: Any,
    ) -> float64:
        clf = class_(**params)
        try:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                evaluation_fn = self.gec_hyperparameters_["score_evaluation_method"]
                maximize_score = self.gec_hyperparameters_["maximize_score"]
                cross_val_scores = cross_val_score(
                    clf,
                    X,
                    y,
                    cv=5,
                    fit_params=self.gec_fit_params_,
                    scoring=evaluation_fn,
                )
                if maximize_score:
                    score = np.mean(cross_val_scores)
                else:
                    score = -np.mean(cross_val_score)
        except Exception as e:
            params_clean = {k: v for k, v in params.items() if v is not None}
            warnings.warn(
                f"Could not calculate cross val scores for parameters: {params_clean}, due to {e}"
            )
            score = np.nan

        return score

    def _optimise_hyperparameters(
        self,
        n_iter: int,
        X: ndarray,
        Y: ndarray,
        **kwargs,
    ) -> Tuple[Dict[str, Optional[Union[int, float, str]]], float64]:

        if self.gec_hyperparameters_["randomize"]:
            np.random.seed(int(datetime.now().timestamp() % 1 * 1e7))

        n_random_exploration = min(
            self.gec_hyperparameters_["n_random_exploration"], int(n_iter / 2)
        )

        for i in tqdm(list(range(n_iter))):
            if (i + self.gec_iter_) < n_random_exploration:
                (
                    selected_arm,
                    selected_combination,
                ) = self._get_random_hyperparameter_configuration()

                mean, sigma = None, None
            else:
                (
                    selected_arm,
                    selected_combination,
                    mean,
                    sigma,
                ) = self._select_parameters()

            arguments = self._build_arguments(selected_arm, selected_combination)

            arguments = self._replace_fixed_args(arguments)
            score = self.score_single_iteration(
                X, Y, self._process_arguments(arguments)
            )

            if np.isnan(score) == False:
                self._update_gec_fields(
                    score, arguments, selected_arm, selected_combination, mean, sigma
                )
            else:
                warnings.warn(f"score is nan for {arguments}")

        return (self.best_params_, self.best_score_gec_)

    def _get_random_hyperparameter_configuration(
        self,
    ) -> Tuple[str_, Tuple[int, float64]]:

        random_arm = np.random.choice(self._categorical_hyperparameter_combinations)
        random_combination = np.array(
            [
                np.random.choice(range_)
                for real_hyperparameter, range_ in self._real_hyperparameters_linear
            ]
        )

        return (
            random_arm,
            random_combination,
        )

    def _select_parameters(self) -> Tuple[str, ndarray, ndarray, ndarray]:

        sampled_reward = np.array(
            [
                beta.rvs(reward["a"], reward["b"])
                for arm, reward in self.rewards_.items()
                if arm in self._categorical_hyperparameter_combinations
            ]
        )
        selected_arm_index = sampled_reward.argmax()
        selected_arm = self._categorical_hyperparameter_combinations[selected_arm_index]

        sets = np.array(
            [
                np.random.choice(range_, self.gec_hyperparameters_["n_sample_initial"])
                for _, range_ in self._real_hyperparameters_linear
            ]
        )

        combinations = self._get_combinations_to_score(sets)

        assert len(combinations), sets

        if len(self.hyperparameter_scores_["inputs"]) > 0:
            self._fit_gaussian()

        mean, sigma = self.gaussian.predict(combinations, return_std=True)

        predicted_rewards = np.array(
            [
                scipy.stats.norm.ppf(
                    self.gec_hyperparameters_["hyperparams_acquisition_percentile"],
                    loc=m,
                    scale=s,
                )
                for m, s in zip(mean, sigma)
            ]
        )

        selected_combination = combinations[np.argmax(predicted_rewards)]

        return (selected_arm, selected_combination, mean, sigma)

    def _get_combinations_to_score(self, sets: ndarray) -> List[ndarray]:
        if len(self.hyperparameter_scores_["inputs"]):
            n_best = max(
                3, int(self.gec_iter_ * self.gec_hyperparameters_["best_share"])
            )
            best_interactions = np.argsort(
                np.array(self.hyperparameter_scores_["output"])
            )[-n_best:]

            best_hyperparameters = np.array(self.hyperparameter_scores_["inputs"])[
                best_interactions, :
            ]

            closest_hyperparameters = cdist(
                best_hyperparameters,
                sets.T,
                metric=self.gec_hyperparameters_["distance_metric"],
            ).argsort(1)[:, : self.gec_hyperparameters_["n_sample"]]
            selected_hyperparameter_indices = np.unique(
                closest_hyperparameters.flatten()
            )

            combinations = list(sets[:, selected_hyperparameter_indices].T)
        else:
            combinations = list(sets[:, : self.gec_hyperparameters_["n_sample"]].T)

        return combinations

    def _update_gec_fields(
        self,
        score: float64,
        arguments: Dict[str, Optional[Union[str, float, int, float64]]],
        selected_arm: Union[str, str_],
        selected_combination: ndarray,
        mean: Optional[ndarray],
        sigma: Optional[ndarray],
    ) -> None:

        self.selected_arms_.append(selected_arm)
        self.hyperparameter_scores_["inputs"].append(
            [float(f) for f in selected_combination]
        )
        self.hyperparameter_scores_["output"].append(score)

        if mean is not None:
            self.hyperparameter_scores_["means"].append(mean)
            self.hyperparameter_scores_["sigmas"].append(sigma)

        if self.best_score_gec_ is not None:
            score_delta = score - self.best_score_gec_
            weighted_score_delta = (
                score_delta * self.gec_hyperparameters_["bandit_greediness"]
            )
            if score_delta > 0:
                self.rewards_[selected_arm]["a"] = (
                    self.rewards_[selected_arm]["a"] + weighted_score_delta
                )
                self.best_score_gec_ = score
                self.best_params_ = self._process_arguments(arguments)
                self.best_params_raw_ = arguments
                self.best_arm_ = selected_arm

            else:
                self.rewards_[selected_arm]["b"] = (
                    self.rewards_[selected_arm]["b"] - weighted_score_delta
                )
        else:
            self.best_score_gec_ = score
            self.best_params_ = self._process_arguments(arguments)
            self.best_params_raw_ = arguments
            self.best_arm_ = selected_arm

    def _replace_fixed_args(
        self, params: Dict[str, Optional[Union[int, float, str]]]
    ) -> Dict[str, Optional[Union[int, float, str]]]:
        if self.fix_boosting_type_:
            params["boosting_type"] = self.retrieve_hyperparameter("boosting_type")

        return params

    def _build_arguments(
        self,
        selected_arm: str,
        real_combination_linear: ndarray,
    ) -> Dict[str, Optional[Union[int, float, str]]]:
        categorical_combination = selected_arm.split("-")
        best_predicted_combination_converted = [
            self._real_hyperparameters_map[name][value]
            for name, value in zip(
                self._real_hyperparameter_names,
                real_combination_linear,
            )
        ]

        hyperparameter_values = categorical_combination + [
            self._cast_to_type(c, t)
            for c, t in zip(
                list(best_predicted_combination_converted),
                self._real_hypermarameter_types,
            )
        ]

        arguments = dict(
            zip(
                list(self._categorical_hyperparameter_names)
                + list(self._real_hyperparameter_names),
                hyperparameter_values,
            )
        )

        return {**self.fixed_params, **self._init_args, **arguments}

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_gaussian(self) -> None:
        output = np.array(self.hyperparameter_scores_["output"])
        output = (output - np.max(output)) + 1
        output[output < -1.0] = -1.0

        self.gaussian.fit(np.array(self.hyperparameter_scores_["inputs"]), output)

    def _get_best_arm(self) -> str:
        mean_reward = np.array(
            [
                reward["a"] / (reward["a"] + reward["b"])
                for arm, reward in self.rewards_.items()
                if arm in self._categorical_hyperparameter_combinations
            ]
        )
        best_arm = self._categorical_hyperparameter_combinations[mean_reward.argmax()]
        return best_arm

    def _find_best_parameters(
        self, step_sizes: List[int] = [16, 8, 4, 2, 1]
    ) -> Dict[str, Optional[Union[int, float, str]]]:

        best_arm = self._get_best_arm()

        self._fit_gaussian()
        sets = [
            list(range_[:: step_sizes[0]]) + [range_[-1]]
            for range_ in self._real_hyperparameter_ranges
        ]
        initial_combinations = np.array(list(itertools.product(*sets)))

        best_combination, _ = self._find_best_parameters_iter(initial_combinations)

        best_params = self._find_best_parameters_from_initial_parameters(
            best_arm, best_combination, step_sizes
        )

        return best_params

    def _find_best_parameters_from_search(
        self, best_arm: str, params: Dict[str, Optional[Union[int, float, str]]]
    ) -> Dict[str, Optional[Union[int, float, str]]]:
        self._fit_gaussian()

        best_params_linear_values = [
            self._real_hyperparameters_map_reverse[name][params[name]]
            for name in self._real_hyperparameter_names
        ]
        best_params = self._find_best_parameters_from_initial_parameters(
            best_arm,
            best_params_linear_values,
            step_sizes=[4, 2, 1],
        )

        return best_params

    def _find_best_parameters_iter(
        self, combinations: ndarray
    ) -> Tuple[ndarray, float64]:

        mean = self.gaussian.predict(combinations)
        best_score = np.max(mean)
        best_combination = combinations[np.argmax(mean)]

        return best_combination, best_score

    def _find_best_parameters_from_initial_parameters(
        self,
        best_arm: str,
        best_combination: Union[List[float16], ndarray],
        step_sizes: List[int],
    ) -> Dict[str, Optional[Union[int, float, str]]]:
        for step_size, previous_step_size in zip(step_sizes[1:], step_sizes[:-1]):

            neighbouring_combinations = self._get_neighbouring_combinations(
                best_combination, step_size, previous_step_size
            )

            best_combination, best_score = self._find_best_parameters_iter(
                neighbouring_combinations
            )

        arguments = self._build_arguments(best_arm, best_combination)

        return arguments

    def _get_neighbouring_combinations(
        self,
        best_combination: Union[List[float16], ndarray],
        step_size: int,
        previous_step_size: int,
    ) -> ndarray:
        new_sets = []
        for real_value, range_ in zip(
            best_combination, self._real_hyperparameter_ranges
        ):
            real_value_index = np.argmax(range_ == real_value)

            start_index = real_value_index - previous_step_size
            start_index = max(start_index, 0)
            end_index = min(real_value_index + previous_step_size, len(range_))

            adjusted_step_size = min(int(len(range_) / 2), step_size)
            new_set = list(range_[start_index:end_index:adjusted_step_size])
            if np.max(new_set) < real_value:
                new_set.append(real_value)

            new_sets.append(new_set)
        neighbouring_combinations = np.array(list(itertools.product(*new_sets)))

        return neighbouring_combinations

    def save_plots(self, path_stem):
        """Create and save plots that summarise GEC trajectory

        Parameters
        ----------
            path_stem : str
                path to folder + file name root to save plots to
        """
        figs = self.plot_gec()
        self._write_figures(figs, path_stem)

    def _write_figures(self, figs, path_stem):
        for plot_name, fig in figs.items():
            fig.savefig(f"{path_stem}_{plot_name}.png")

    def plot_gec(self) -> Dict[str, Figure]:
        """Create figures to summarise GEC trajectory

        Returns
        -------
            figs : dict[str, fig]
                a dictionary of figures
        """

        figs = {}

        fig, axes = plt.subplots(
            nrows=2, ncols=2, sharex=True, sharey=False, figsize=(12, 12)
        )
        ax1, ax2, ax3, ax4 = axes.flatten()

        x = np.arange(len(self.hyperparameter_scores_["means"]))
        self._plot_mean_prediction_and_mean_variance(ax1, x)
        self._plot_prediction_std_and_variance_std(ax2, x)
        self._plot_prediction_mean_variance_correlation(ax3, x)
        self._plot_linear_scaled_parameter_samples(ax4)

        figs["parameters"] = fig

        return figs

    def _plot_mean_prediction_and_mean_variance(self, ax: Axes, x: ndarray) -> None:
        gp_mean_prediction = [np.mean(x) for x in self.hyperparameter_scores_["means"]]
        gp_mean_sigma = [np.mean(x) for x in self.hyperparameter_scores_["sigmas"]]

        ax.plot(x, gp_mean_prediction, label="mean_prediction")
        ax.plot(x, gp_mean_sigma, label="mean_sigma")
        ax.legend(loc="upper right")

    def _plot_prediction_std_and_variance_std(self, ax: Axes, x: ndarray) -> None:
        gp_prediction_variance = [
            np.std(x) for x in self.hyperparameter_scores_["means"]
        ]
        gp_sigma_variance = [np.std(x) for x in self.hyperparameter_scores_["sigmas"]]

        ax.plot(x, gp_prediction_variance, label="prediction_variance")
        ax.plot(x, gp_sigma_variance, label="sigma_variance")
        ax.legend(loc="lower right")

    def _plot_prediction_mean_variance_correlation(self, ax: Axes, x: ndarray) -> None:
        correlation = [
            np.corrcoef(
                self.hyperparameter_scores_["means"][i],
                self.hyperparameter_scores_["sigmas"][i],
            )[0, 1]
            for i in x
        ]

        ax.plot(
            x,
            correlation,
            label="corr(preds mean, preds variance)",
        )
        ax.legend(loc="lower right")

    def _plot_linear_scaled_parameter_samples(self, ax: Axes) -> None:
        inputs_ = np.array(self.hyperparameter_scores_["inputs"])
        x = np.arange(inputs_.shape[0])

        for i in range(inputs_.shape[1]):
            ax.plot(x, inputs_[:, i], label=self._real_hyperparameter_names[i])
