import inspect
import itertools
from typing import Optional, Union, Dict, Callable

import numpy as np

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

        self.hyperparameters = [
            [("boosting", ["gbdt"])],
            [
                ("lambda_l1", list((np.logspace(0.00, 1, 50) - 1) / 9)),
                ("num_leaves", [int(x) for x in np.arange(10, 200, 1)]),
                ("min_data_in_leaf", [int(x) for x in np.arange(2, 50, 2)]),
                ("feature_fraction", [float(x) for x in np.arange(0.1, 1.1, 0.1)]),
                ("learning_rate", list((np.logspace(0.01, 1.5, 30)) / 100)),
            ],
        ]
        self.kernel = RBF(1.0)
        self.gp_datas = None

    def fit(self, X, y, n_iter=100):

        (best_configuration, best_score), gp_datas = self.optimise_hyperparameters(
            self.hyperparameters,
            n_iter,
            X,
            y,
            20,
            gp_datas=self.gp_datas,
            kernel=self.kernel,
        )

        gec = GEC(**best_configuration)
        self.__dict__.update(gec.__dict__)

        if hasattr(self, "gec_iter"):
            self.gec_iter += n_iter
        else:
            self.gec_iter = n_iter

        if not hasattr(self, "best_score") or best_score > self.best_score:
            self.best_score = best_score
            self.best_configuration = best_configuration

        self.gp_datas = gp_datas

        super().fit(X, y)

        return self

    @classmethod
    def optimise_hyperparameters(
        cls, hyperparameters, n_iter, X, Y, n_sample=10, gp_datas=None, **kwargs
    ):
        categorical_hyperparameters = [
            "-".join(y) for y in itertools.product(*[x[1] for x in hyperparameters[0]])
        ]
        ranges = [x[1] for x in hyperparameters[1]]
        gaussian = GaussianProcessRegressor(**kwargs)
        # parameters for gaussian process
        if gp_datas is not None:
            assert np.all(
                np.array(sorted(list(gp_datas.keys())))
                == np.array(categorical_hyperparameters)
            )
        else:
            gp_datas = {
                c: (np.zeros((0, len(ranges))), np.zeros((0)))
                for c in categorical_hyperparameters
            }

        best_score = None
        best_configuration = None

        # parameters for bandit
        counts = {c: 0.001 for c in categorical_hyperparameters}
        rewards = {c: [1] for c in categorical_hyperparameters}

        for i in range(n_iter):
            ucb = np.array(
                [
                    np.mean(rewards[c]) * 10
                    + np.sqrt(2 * np.sum(list(counts.values())) / count)
                    for c, count in counts.items()
                ]
            )
            selected_arm_index = ucb.argmax()
            selected_arm = categorical_hyperparameters[selected_arm_index]
            counts[selected_arm] = int(counts[selected_arm] + 1)

            if gp_datas[selected_arm][0].shape[0] > 0:
                gaussian.fit(gp_datas[selected_arm][0], gp_datas[selected_arm][1])

            sets = [np.random.choice(range_, n_sample) for range_ in ranges]
            sets_types = [s.dtype for s in sets]
            combinations = cartesian(sets)

            mean, sigma = gaussian.predict(combinations, return_std=True)

            predicted_rewards = np.array(
                [np.random.normal(m, s) for m, s in zip(mean, sigma)]
            )

            hyperparameter_values = selected_arm.split("-") + [
                cls.cast_to_type(c, t)
                for c, t in zip(
                    combinations[np.argmax(predicted_rewards)].tolist(), sets_types
                )
            ]
            arguments = dict(
                zip(
                    [x[0] for x in hyperparameters[0]]
                    + [x[0] for x in hyperparameters[1]],
                    hyperparameter_values,
                )
            )
            clf = LGBMClassifier(**arguments)

            score = np.mean(cross_val_score(clf, X, Y, cv=5))
            if np.isnan(score):
                score = 0

            if best_score is None or score > best_score:
                best_score = score
                best_configuration = arguments

            gp_datas[selected_arm] = (
                np.concatenate(
                    [
                        gp_datas[selected_arm][0],
                        combinations[np.argmax(predicted_rewards)].reshape((1, -1)),
                    ],
                    0,
                ),
                np.concatenate([gp_datas[selected_arm][1], [score]]),
            )
            rewards[selected_arm] += [score]

            if np.sum(np.array(rewards[selected_arm]) == 0) > 1:
                failure = selected_arm
                print(failure)
                counts.pop(failure)
                rewards.pop(failure)
                gp_datas.pop(failure)
                categorical_hyperparameters = [
                    hp for hp in categorical_hyperparameters if hp != failure
                ]

        return ((best_configuration, best_score), gp_datas)

    @classmethod
    def cast_to_type(cls, value, type_):
        if type_ == np.float64:
            return float(value)
        elif type_ == np.int64:
            return int(value)
        else:
            raise Exception(f"type {type_} currently not supported")
