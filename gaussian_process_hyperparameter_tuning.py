import numpy as np
from sklearn.utils.extmath import cartesian
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score

"""
hyperparameters = [
    [("max_features", ['auto', 'sqrt', 'log2']),
    ("criterion", ['gini','entropy'])],
    
    [("n_estimators", np.arange(1, 1000, 1)),
    ("max_depth", np.arange(1, 1000, 1)),
    ("min_samples_split", np.arange(2, 100, 1)),
    ("min_samples_leaf", np.arange(1, 50, 1)),
    ("min_impurity_decrease", np.arange(0.0, 1.0, 0.01))]
]

optimise_hpyerparameters(RandomForestClassifier, hyperparameters, 100, X, Y, n_sample)
"""


def cast_to_type(value, type_):
    if type_ == np.float64:
        return float(value)
    elif type_ == np.int64:
        return int(value)
    else:
        raise Exception(f"type {type_} currently not supported")


def optimise_hyperparameters(
    Class, hyperparameters, n_iter, X, Y, n_sample=10, gp_datas=None, **kwargs
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
            cast_to_type(c, t)
            for c, t in zip(
                combinations[np.argmax(predicted_rewards)].tolist(), sets_types
            )
        ]
        arguments = dict(
            zip(
                [x[0] for x in hyperparameters[0]] + [x[0] for x in hyperparameters[1]],
                hyperparameter_values,
            )
        )
        clf = Class(**arguments)

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
