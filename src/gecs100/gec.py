import inspect
import itertools
from typing import Optional, Union, Dict, Callable
import os
import contextlib
from tqdm import tqdm

import warnings
import numpy as np
import json
import math
import copy


import matplotlib.pyplot as plt

from scipy.stats import beta
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.extmath import cartesian
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from lightgbm.compat import SKLEARN_INSTALLED


class GEC(LGBMClassifier):
    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[str, Callable]] = None,
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
        n_jobs: int = -1,
        silent: Union[bool, str] = "warn",
        importance_type: str = "split",
        **kwargs,
    ):
        assert str(inspect.signature(LGBMClassifier.__init__)) == str(
            inspect.signature(GEC.__init__)
        )
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : str, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
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
            Minimum sum of instance weight (hessian) needed in a child (leaf).
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
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
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
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            group : array-like
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of y_pred for each sample point.
            hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of y_pred for each sample point.

        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
        and you should group grad and hess in this way as well.
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
        self.silent = silent
        self.importance_type = importance_type
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight = None
        self._class_map = None
        self._n_features = None
        self._n_features_in = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

        self.categorical_hyperparameters = [
            ("boosting", ["gbdt", "dart", "rf"]),
            ("bagging", ["yes_bagging", "no_bagging"]),
        ]

        self.categorical_hyperparameter_names, _ = zip(
            *self.categorical_hyperparameters
        )

        prohibited_combinations = ["rf-no_bagging"]
        self.categorical_hyperparameter_combinations = [
            "-".join(y)
            for y in itertools.product(
                *[x[1] for x in self.categorical_hyperparameters]
            )
            if "-".join(y) not in prohibited_combinations
        ]

        ten_to_thousand = np.concatenate(
            [
                np.arange(10, 100, 10),
                np.arange(100, 200, 20),
                np.arange(200, 500, 50),
                np.arange(500, 1001, 100),
            ]
        )
        self.real_hyperparameters = [
            ("num_leaves", np.arange(10, 200, 1)),
            ("learning_rate", (np.logspace(0.001, 1.5, 150)) / 100),
            (
                "n_estimators",
                ten_to_thousand[:10],
            ),
            ("max_bin", ten_to_thousand),
            ("max_depth", np.concatenate([np.array([-1]), ten_to_thousand[0:21:3]])),
            ("lambda_l1", (np.logspace(0.00, 1, 100) - 1) / 9),
            ("lambda_l2", (np.logspace(0.00, 1, 100) - 1) / 9),
            ("min_data_in_leaf", np.arange(2, 50, 1)),
            (
                "feature_fraction",
                np.concatenate([np.arange(0.1, 1.00, 0.01), np.array([1.0])]),
            ),
        ]
        self.sampling_probabilities = {}
        self.real_hyperparameters_linear = [
            (name, np.arange(-1, 1, 2 / len(values)).astype(np.float16))
            for name, values in self.real_hyperparameters
        ]

        self.real_hyperparameters_map = {
            name: dict(zip(linear_values, real_values))
            for ((name, linear_values), (_, real_values)) in zip(
                self.real_hyperparameters_linear, self.real_hyperparameters
            )
        }

        self.real_hyperparameters_map_reverse = {
            name: dict(zip(real_values, linear_values))
            for ((name, linear_values), (_, real_values)) in zip(
                self.real_hyperparameters_linear, self.real_hyperparameters
            )
        }

        self.real_hyperparameter_names, self.linear_ranges = zip(
            *self.real_hyperparameters_linear
        )
        self.real_hyperparameter_name_to_index = {
            n: i for i, n in enumerate(self.real_hyperparameter_names)
        }

        self.sets_types = [np.array(s).dtype for _, s in self.real_hyperparameters]

        self.kernel = RBF(1.0)
        self.gaussian = GaussianProcessRegressor(kernel=self.kernel)
        self.gp_datas = {
            c: {"inputs": [], "output": [], "means": [], "sigmas": []}
            for c in self.categorical_hyperparameter_combinations
        }
        self.gaussian_bagging = GaussianProcessRegressor(kernel=self.kernel)
        self.bagging_datas = {
            c: {"inputs": [], "output": [], "means": [], "sigmas": []}
            for c in self.categorical_hyperparameter_combinations
        }
        self.bagging_combinations = list(
            itertools.product(
                *[
                    np.arange(1, 11, 1),
                    np.arange(0.05, 1.0, 0.05),
                ]
            )
        )

        self.best_score = None
        self.best_params_ = None
        self.n_sample = 1000
        self.n_iterations = 0

        self.last_score = None
        # parameters for bandit
        self.rewards = {
            c: {"a": 1, "b": 1} for c in self.categorical_hyperparameter_combinations
        }
        self.selected_arms = []

    def serialise(self, path):
        gp_datas = {
            k: {
                k2: [list(vv) if isinstance(vv, np.ndarray) else vv for vv in v]
                for k2, v in values.items()
            }
            for k, values in self.gp_datas.items()
        }

        representation = {
            "gp_datas": gp_datas,
            "gec_iter": self.gec_iter,
            "best_params_": self.best_params_,
            "best_score": self.best_score,
            "best_params_grid": self.best_params_grid,
            "best_score_grid": self.best_score_grid,
            "rewards": self.rewards,
            "selected_arms": self.selected_arms,
        }
        with open(path, "w") as f:
            f.write(json.dumps(representation))

    def load_state(self, path, X=None, y=None):
        with open(path, "r") as f:
            representation = json.loads(f.read())

        best_params_grid = representation["best_params_grid"]

        if X is not None and y is not None:
            gec = GEC(**{**best_params_grid, "random_state": 101})
            self.__dict__.update(gec.__dict__)
            super().fit(X, y)
        else:
            warnings.warn(
                "If X and y are not provided, the GEC model is not fitted for inference"
            )

        self.gp_datas = {
            k: {
                k2: [np.array(vv) if isinstance(vv, list) else vv for vv in v]
                for k2, v in values.items()
            }
            for k, values in representation["gp_datas"].items()
        }

        self.gec_iter = int(representation["gec_iter"])
        self.best_params_ = representation["best_params_"]
        self.best_score = float(representation["best_score"])
        self.best_params_grid = best_params_grid
        self.best_score_grid = float(representation["best_score_grid"])
        self.rewards = representation["rewards"]
        self.selected_arms = representation["selected_arms"]

    def summarise_gp_datas(self):

        figs = {}

        for categorical_combination in self.categorical_hyperparameter_combinations:
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
            ax1, ax2, ax3, ax4 = axes.flatten()

            x = np.arange(len(self.gp_datas[categorical_combination]["means"]))
            self.plot_mean_prediction_and_mean_variance(categorical_combination, ax1, x)
            self.plot_prediction_std_and_variance_std(categorical_combination, ax2, x)
            self.plot_prediction_mean_variance_correlation(
                categorical_combination, ax3, x
            )
            self.plot_linear_scaled_parameter_samples(categorical_combination, ax4, x)
            figs[categorical_combination] = fig

        return figs

    def write_figures(self, figs, path_stem):
        for categorical_combination, fig in figs.items():
            fig.savefig(f"{path_stem}_{categorical_combination}.png")

    def save_figs(self, path_stem):
        figs = self.summarise_gp_datas()
        self.write_figures(figs, path_stem)

    def plot_linear_scaled_parameter_samples(self, cat, ax, x):
        inputs_ = np.array(self.gp_datas[cat]["inputs"])
        for i in range(inputs_.shape[1]):
            ax.plot(x, inputs_[:, i], label=self.real_hyperparameter_names[i])

    def plot_prediction_mean_variance_correlation(self, cat, ax, x):
        correlation = [
            np.corrcoef(
                self.gp_datas[cat]["means"][i], self.gp_datas[cat]["sigmas"][i]
            )[0, 1]
            for i in x
        ]

        ax.plot(
            x,
            correlation,
            label="corr(preds mean, preds variance)",
        )
        ax.legend(loc="lower right")

    def plot_prediction_std_and_variance_std(self, cat, ax, x):
        gp_prediction_variance = [np.std(x) for x in self.gp_datas[cat]["means"]]
        gp_sigma_variance = [np.std(x) for x in self.gp_datas[cat]["sigmas"]]

        ax.plot(x, gp_prediction_variance, label="prediction_variance")
        ax.plot(x, gp_sigma_variance, label="sigma_variance")
        ax.legend(loc="lower right")

    def plot_mean_prediction_and_mean_variance(self, cat, ax, x):
        gp_mean_prediction = [np.mean(x) for x in self.gp_datas[cat]["means"]]
        gp_mean_sigma = [np.mean(x) for x in self.gp_datas[cat]["sigmas"]]

        ax.plot(x, gp_mean_prediction, label="mean_prediction")
        ax.plot(x, gp_mean_sigma, label="mean_sigma")
        ax.legend(loc="upper right")

    def find_best_parameters(self, step_sizes=[16, 8, 4, 2, 1]):

        sets = [
            list(range_[:: step_sizes[0]]) + [range_[-1]]
            for range_ in self.linear_ranges
        ]
        real_combinations = np.array(list(itertools.product(*sets)))

        n_selected_arms = int(len(self.selected_arms) * 0.3)
        arms, counts = np.unique(
            self.selected_arms[n_selected_arms:],
            return_counts=True,
        )

        top_3 = arms[np.argsort(counts)][-3:]

        initial_combinations = {
            categorical_combination: real_combinations
            for categorical_combination in top_3
        }

        best_combinations, _ = self.find_best_parameters_iter(initial_combinations)

        return self.find_best_parameters_from_initial_parameters(
            best_combinations, step_sizes
        )

    def find_best_parameters_from_initial_parameters(
        self, best_combinations, step_sizes
    ):
        for step_size, previous_step_size in zip(step_sizes[1:], step_sizes[:-1]):

            neighbouring_combinations = self.get_neighbouring_combinations(
                best_combinations, step_size, previous_step_size
            )

            new_ranges = list(
                zip(
                    self.real_hyperparameter_names,
                    list(neighbouring_combinations.values())[0].min(0),
                    list(neighbouring_combinations.values())[0].max(0),
                )
            )

            best_combinations, best_scores = self.find_best_parameters_iter(
                neighbouring_combinations
            )

        max_score = np.max(list(best_scores.values()))
        for categorical_combination, real_combination in best_combinations.items():
            if best_scores[categorical_combination] == max_score:
                arm_best_score = str(categorical_combination)
                arguments = self.build_arguments(
                    categorical_combination.split("-"), real_combination
                )

        if "yes_bagging" in arm_best_score:
            bagging_data = np.array(self.bagging_datas[arm_best_score]["inputs"])
            bagging_data[:, 0] = bagging_data[:, 0] / 10
            self.gaussian_bagging.fit(
                bagging_data,
                np.array(self.bagging_datas[arm_best_score]["output"])
                - self.adjustment_factor,
            )
            mean_bagging = self.gaussian_bagging.predict(self.bagging_combinations)
            best_predicted_combination_bagging = self.bagging_combinations[
                np.argmax(mean_bagging)
            ]

            (
                arguments["bagging_freq"],
                arguments["bagging_fraction"],
            ) = best_predicted_combination_bagging

        del arguments["bagging"]

        return arguments

    def get_neighbouring_combinations(
        self, best_combinations, step_size, previous_step_size
    ):
        neighbouring_combinations = {}
        for categorical_combination, real_combination in best_combinations.items():
            new_sets = []
            for real_value, range_ in zip(real_combination, self.linear_ranges):
                real_value_index = np.argmax(range_ == real_value)

                start_index = real_value_index - previous_step_size
                start_index = start_index if start_index > 0 else 0
                end_index = min(real_value_index + previous_step_size, len(range_))

                adjusted_step_size = min(int(len(range_) / 2), step_size)
                new_set = list(range_[start_index:end_index:adjusted_step_size])
                if np.max(new_set) < real_value:
                    new_set.append(real_value)

                new_sets.append(new_set)

            neighbouring_combinations[categorical_combination] = np.array(
                list(itertools.product(*new_sets))
            )

        return neighbouring_combinations

    def find_best_parameters_iter(self, combinations):
        best_combinations = {}
        best_scores = {}
        for categorical_combination, combs in combinations.items():
            self.gaussian.fit(
                self.gp_datas[categorical_combination]["inputs"],
                self.gp_datas[categorical_combination]["output"],
            )

            mean = self.gaussian.predict(combs)
            best_scores[categorical_combination] = np.max(mean)
            best_combination = combs[np.argmax(mean)]
            best_combinations[categorical_combination] = best_combination

        return best_combinations, best_scores

    def find_best_parameters_from_search(self, params):

        if "bagging_freq" in params:
            del params["bagging_freq"]
            del params["bagging_fraction"]
            bagging = "yes_bagging"
        else:
            bagging = "no_bagging"
        boosting = params.pop("boosting")
        categorical_combination = f"{boosting}-{bagging}"

        best_params_linear_values = [
            self.real_hyperparameters_map_reverse[name][params[name]]
            for name in self.real_hyperparameter_names
        ]
        best_params = self.find_best_parameters_from_initial_parameters(
            {categorical_combination: best_params_linear_values},
            step_sizes=[4, 2, 1],
        )
        return best_params

    def calculate_empirical_score(self, X, y, params):
        clf = LGBMClassifier(**params)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            score = np.mean(cross_val_score(clf, X, y, cv=3))
        return score

    def fit(self, X, y, n_iter=100):

        self.adjustment_factor = 1 / len(np.unique(y))  # get mean closer to 0

        self.best_scores_gec = {}
        self.best_params_gec = {}
        (
            self.best_params_gec["search"],
            self.best_scores_gec["search"],
        ) = self.optimise_hyperparameters(
            n_iter, X, y, self.best_score, self.best_params_
        )
        self.best_params_gec["grid"] = self.find_best_parameters()
        self.best_scores_gec["grid"] = self.calculate_empirical_score(
            X, y, self.best_params_gec["grid"]
        )
        best_params_prep = copy.deepcopy(self.best_params_gec["search"])
        self.best_params_gec[
            "grid_from_search"
        ] = self.find_best_parameters_from_search(best_params_prep)

        self.best_scores_gec["grid_from_search"] = self.calculate_empirical_score(
            X, y, self.best_params_gec["grid_from_search"]
        )

        for source, score in self.best_scores_gec.items():
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_params_ = self.best_params_gec[source]

        self.gec_iter = np.sum(
            [len(value["output"]) for value in self.gp_datas.values()]
        )

        # gp_datas, rewards = copy.deepcopy(self.gp_datas), copy.deepcopy(self.rewards)
        # selected_arms = copy.deepcopy(self.selected_arms)
        gec = GEC(**{**self.best_params_, "random_state": 101})

        for k, v in gec.__dict__.items():
            if k not in self.__dict__ or self.__dict__[k] is None:
                self.__dict__[k] = v

        super().fit(X, y)

        # self.gp_datas, self.rewards = gp_datas, rewards
        # self.selected_arms = selected_arms

        return self

    def build_arguments(self, categorical_combination, real_combination_linear):
        best_predicted_combination_converted = [
            self.real_hyperparameters_map[name][value]
            for name, value in zip(
                self.real_hyperparameter_names,
                real_combination_linear,
            )
        ]

        hyperparameter_values = categorical_combination + [
            self.cast_to_type(c, t)
            for c, t in zip(list(best_predicted_combination_converted), self.sets_types)
        ]

        arguments = dict(
            zip(
                self.categorical_hyperparameter_names + self.real_hyperparameter_names,
                hyperparameter_values,
            )
        )
        return arguments

    def optimise_hyperparameters(
        self,
        n_iter,
        X,
        Y,
        best_score,
        best_params,
        **kwargs,
    ):

        # parameters for gaussian process
        assert np.all(
            np.array(sorted(list(self.gp_datas.keys())))
            == np.array(sorted(self.categorical_hyperparameter_combinations))
        )

        for i in tqdm(list(range(n_iter))):
            sampled_reward = np.array(
                [
                    beta.rvs(reward["a"], reward["b"])
                    for _, reward in self.rewards.items()
                ]
            )
            selected_arm_index = sampled_reward.argmax()
            selected_arm = self.categorical_hyperparameter_combinations[
                selected_arm_index
            ]
            self.selected_arms.append(selected_arm)

            sets = [
                list(
                    np.random.choice(
                        range_,
                        self.n_sample,
                        p=self.sampling_probabilities.get(real_hyperparameter, None),
                    )
                )
                for real_hyperparameter, range_ in self.real_hyperparameters_linear
            ]

            combinations = [np.array(comb) for comb in zip(*sets)]
            assert len(combinations), sets

            if len(self.gp_datas[selected_arm]["inputs"]) > 0:
                self.gaussian.fit(
                    np.array(self.gp_datas[selected_arm]["inputs"]),
                    np.array(self.gp_datas[selected_arm]["output"])
                    - self.adjustment_factor,
                )

            mean, sigma = self.gaussian.predict(combinations, return_std=True)

            predicted_rewards = np.array(
                [m + 0.3 * np.random.normal(m, s) for m, s in zip(mean, sigma)]
            )

            best_predicted_combination = combinations[np.argmax(predicted_rewards)]
            arguments = self.build_arguments(
                selected_arm.split("-"), best_predicted_combination
            )

            if "yes_bagging" in selected_arm:
                if len(self.bagging_datas[selected_arm]["inputs"]) > 0:
                    self.gaussian_bagging.fit(
                        np.array(self.bagging_datas[selected_arm]["inputs"]),
                        np.array(self.bagging_datas[selected_arm]["output"])
                        - self.adjustment_factor,
                    )
                mean_bagging, sigma_bagging = self.gaussian_bagging.predict(
                    self.bagging_combinations, return_std=True
                )
                predicted_rewards_bagging = np.array(
                    [
                        m + 0.3 * np.random.normal(m, s)
                        for m, s in zip(mean_bagging, sigma_bagging)
                    ]
                )
                best_predicted_combination_bagging = self.bagging_combinations[
                    np.argmax(predicted_rewards_bagging)
                ]
                (
                    arguments["bagging_freq"],
                    arguments["bagging_fraction"],
                ) = best_predicted_combination_bagging

                assert arguments["bagging_freq"] > arguments["bagging_fraction"]
            del arguments["bagging"]

            arguments["verbosity"] = -1

            clf = LGBMClassifier(**arguments)

            try:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    score = np.mean(cross_val_score(clf, X, Y, cv=3))

                if np.isnan(score):
                    score = 0

                if best_score is None or score > best_score:
                    best_score = score
                    best_params = arguments

                self.gp_datas[selected_arm]["inputs"].append(best_predicted_combination)
                self.gp_datas[selected_arm]["output"].append(score)
                self.gp_datas[selected_arm]["means"].append(mean)
                self.gp_datas[selected_arm]["sigmas"].append(sigma)

                if "bagging_freq" in arguments:
                    self.bagging_datas[selected_arm]["inputs"].append(
                        best_predicted_combination_bagging
                    )
                    self.bagging_datas[selected_arm]["output"].append(score)
                    self.bagging_datas[selected_arm]["means"].append(mean_bagging)
                    self.bagging_datas[selected_arm]["sigmas"].append(sigma_bagging)

                if self.last_score is not None:
                    score_delta = score - self.last_score
                    if score_delta > 0:
                        self.rewards[selected_arm]["a"] = (
                            self.rewards[selected_arm]["a"] + score_delta
                        )
                    else:
                        self.rewards[selected_arm]["b"] = (
                            self.rewards[selected_arm]["b"] - score_delta
                        )
                self.last_score = score

            except Exception as e:
                warnings.warn(f"These arguments led to an Error: {arguments}: {e}")

        return (best_params, best_score)

    def tested_parameter_combinations(self):
        real_hyperparameter_names, _ = zip(*self.real_hyperparameters_linear)

        gp_datas_parameters = {}

        for categorical_hyperparameter_combination, (
            parameter_keys,
            reward,
        ) in self.gp_datas.items():
            parameters = [
                [
                    self.real_hyperparameters_map[name][value]
                    for name, value in zip(real_hyperparameter_names, pars)
                ]
                for pars in parameter_keys
            ]
            gp_datas_parameters[categorical_hyperparameter_combination] = (
                parameters,
                reward,
            )
        return gp_datas_parameters

    @classmethod
    def cast_to_type(cls, value, type_):
        if type_ == np.float64:
            return float(value)
        elif type_ == np.int64:
            return int(value)
        else:
            raise Exception(f"type {type_} currently not supported")
