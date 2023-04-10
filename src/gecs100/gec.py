import inspect
import itertools
from typing import Optional, Union, Dict, Callable

import numpy as np
import json
import math

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

        self.categorical_hyperparameters = [("boosting", ["gbdt"])]

        self.categorical_hyperparameter_names, _ = zip(
            *self.categorical_hyperparameters
        )

        self.categorical_hyperparameter_combinations = [
            "-".join(y)
            for y in itertools.product(
                *[x[1] for x in self.categorical_hyperparameters]
            )
        ]

        self.real_hyperparameters = [
            ("lambda_l1", (np.logspace(0.00, 1, 100) - 1) / 9),
            ("num_leaves", [int(x) for x in np.arange(10, 200, 1)]),
            ("min_data_in_leaf", [int(x) for x in np.arange(2, 50, 1)]),
            ("feature_fraction", [float(x) for x in np.arange(0.1, 1.01, 0.01)]),
            ("learning_rate", (np.logspace(0.001, 1.5, 150)) / 100),
        ]
        self.real_hyperparameters_linear = [
            (name, np.arange(-1, 1, 2 / len(values)))
            for name, values in self.real_hyperparameters
        ]

        self.real_hyperparameters_map = {
            name: dict(zip(linear_values, real_values))
            for ((name, linear_values), (_, real_values)) in zip(
                self.real_hyperparameters_linear, self.real_hyperparameters
            )
        }

        self.real_hyperparameter_names, self.linear_ranges = zip(
            *self.real_hyperparameters_linear
        )
        self.sets_types = [np.array(s).dtype for _, s in self.real_hyperparameters]

        self.kernel = RBF(0.02)
        self.gaussian = GaussianProcessRegressor(kernel=self.kernel)
        self.gp_datas = None
        self.best_score = None
        self.best_params_ = None
        self.n_sample = 1000
        self.n_iterations = 0

    def export_gp_datas(self, path):
        gp_datas = {
            k: {
                k2: [list(vv) if isinstance(vv, np.ndarray) else vv for vv in v]
                for k2, v in values.items()
            }
            for k, values in self.gp_datas.items()
        }
        with open(path, "w") as f:
            f.write(json.dumps(gp_datas))

    def load_gp_datas(self, path):
        with open(path, "r") as f:
            gp_datas = json.loads(f.read())

        self.gp_datas = {
            k: {
                k2: [np.array(vv) if isinstance(vv, list) else vv for vv in v]
                for k2, v in values.items()
            }
            for k, values in gp_datas.items()
        }

    def find_best_parameters(self):
        sets = [
            list(range_[:: math.floor(len(range_) / 10)])
            for range_ in self.linear_ranges
        ]
        real_combinations = np.array(list(itertools.product(*sets)))
        initial_combinations = {
            categorical_param_comb: real_combinations
            for categorical_param_comb in self.categorical_hyperparameter_combinations
        }
        best_combinations, _ = self.find_best_parameters_iter(initial_combinations)
        neighbouring_combinations = self.get_neghbouring_combinations(best_combinations)

        best_combinations_2, best_scores_2 = self.find_best_parameters_iter(
            neighbouring_combinations
        )
        max_score = np.max(list(best_scores_2.values()))
        for categorical_combination, real_combination in best_combinations_2.items():
            if best_scores_2[categorical_combination] == max_score:
                arguments = self.build_arguments(
                    categorical_combination.split("-"), real_combination
                )
        return (arguments, max_score)

    def get_neghbouring_combinations(self, best_combinations):

        neighbouring_combinations = {}
        for categorical_combination, real_combination in best_combinations.items():
            new_sets = []
            for real_value, range_ in zip(real_combination, self.linear_ranges):
                real_value_index = np.argmax(range_ == real_value)
                step_size = math.floor(len(range_) / 10)
                start_index = real_value_index - step_size
                start_index = start_index if start_index > 0 else 0
                end_index = min(real_value_index + step_size, len(range_))
                new_set = list(range_[start_index:end_index:3])
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

    def fit(self, X, y, n_iter=100):

        (best_params, best_score), gp_datas = self.optimise_hyperparameters(
            n_iter, X, y, self.gp_datas, self.best_score, self.best_params_
        )

        best_params_grid, best_score_grid = self.find_best_parameters()

        gec = GEC(**best_params_grid)
        self.__dict__.update(gec.__dict__)

        if hasattr(self, "gec_iter"):
            self.gec_iter += n_iter
        else:
            self.gec_iter = n_iter

        self.best_params_grid, self.best_score_grid = best_params_grid, best_score_grid

        if self.best_score is None or best_score > self.best_score:
            self.best_score = best_score
            self.best_params_ = best_params

        self.gp_datas = gp_datas

        super().fit(X, y)

        self.n_iterations = np.sum(
            [len(value["output"]) for value in self.gp_datas.values()]
        )

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
        gp_datas,
        best_score,
        best_params,
        **kwargs,
    ):

        # parameters for gaussian process
        if gp_datas is not None:
            assert np.all(
                np.array(sorted(list(gp_datas.keys())))
                == np.array(self.categorical_hyperparameter_combinations)
            )
        else:
            gp_datas = {
                c: {"inputs": [], "output": [], "means": [], "sigmas": []}
                for c in self.categorical_hyperparameter_combinations
            }

        # parameters for bandit
        counts = {c: 0.001 for c in self.categorical_hyperparameter_combinations}
        rewards = {c: [1] for c in self.categorical_hyperparameter_combinations}

        for i in range(n_iter):
            ucb = np.array(
                [
                    np.mean(rewards[c]) * 10
                    + np.sqrt(2 * np.sum(list(counts.values())) / count)
                    for c, count in counts.items()
                ]
            )
            selected_arm_index = ucb.argmax()
            selected_arm = self.categorical_hyperparameter_combinations[
                selected_arm_index
            ]
            counts[selected_arm] = int(counts[selected_arm] + 1)

            if len(gp_datas[selected_arm]["inputs"]) > 0:
                adjustment_factor = 1 / len(np.unique(Y))
                self.gaussian.fit(
                    np.array(gp_datas[selected_arm]["inputs"]),
                    np.array(gp_datas[selected_arm]["output"]) - adjustment_factor,
                )

            sets = [
                list(np.random.choice(range_, self.n_sample))
                for range_ in self.linear_ranges
            ]
            combinations = [np.array(comb) for comb in zip(*sets)]

            mean, sigma = self.gaussian.predict(combinations, return_std=True)

            predicted_rewards = np.array(
                [m + 0.3 * np.random.normal(m, s) for m, s in zip(mean, sigma)]
            )

            best_predicted_combination = combinations[np.argmax(predicted_rewards)]
            arguments = self.build_arguments(
                selected_arm.split("-"), best_predicted_combination
            )

            clf = LGBMClassifier(**arguments)

            score = np.mean(cross_val_score(clf, X, Y, cv=5))
            if np.isnan(score):
                score = 0

            if best_score is None or score > best_score:
                best_score = score
                best_params = arguments

            gp_datas[selected_arm]["inputs"].append(best_predicted_combination)
            gp_datas[selected_arm]["output"].append(score)
            gp_datas[selected_arm]["means"].append(mean)
            gp_datas[selected_arm]["sigmas"].append(sigma)

            rewards[selected_arm] += [score]

            if np.sum(np.array(rewards[selected_arm]) == 0) > 1:
                failure = selected_arm
                print(failure)
                counts.pop(failure)
                rewards.pop(failure)
                gp_datas.pop(failure)
                self.categorical_hyperparameter_combinations = [
                    hp
                    for hp in self.categorical_hyperparameter_combinations
                    if hp != failure
                ]

        return ((best_params, best_score), gp_datas)

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
