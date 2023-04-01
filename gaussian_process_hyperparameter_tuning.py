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


def optimise_hyperparameters(
    Class, hyperparameters, n_iter, X, Y, n_sample=10, **kwargs
):
    categorical_hyperparameters = [
        "-".join(y) for y in itertools.product(*[x[1] for x in hyperparameters[0]])
    ]
    ranges = [x[1] for x in hyperparameters[1]]
    gaussian = GaussianProcessRegressor(**kwargs)
    gp_datas = {
        c: (np.zeros((0, len(ranges))), np.zeros((0)))
        for c in categorical_hyperparameters
    }

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
        selected_arm = categorical_hyperparameters[ucb.argmax()]
        counts[selected_arm] = int(counts[selected_arm] + 1)

        if gp_datas[selected_arm][0].shape[0] > 0:
            gaussian.fit(gp_datas[selected_arm][0], gp_datas[selected_arm][1])

        sets = [np.random.choice(range_, n_sample) for range_ in ranges]
        combinations = cartesian(sets)

        mean, sigma = gaussian.predict(combinations, return_std=True)

        predicted_rewards = np.array(
            [np.random.normal(m, s) for m, s in zip(mean, sigma)]
        )

        hyperparameter_values = (
            selected_arm.split("-")
            + combinations[np.argmax(predicted_rewards)].tolist()
        )
        arguments = zip(
            [x[0] for x in hyperparameters[0]] + [x[0] for x in hyperparameters[1]],
            hyperparameter_values,
        )
        clf = Class(**dict(arguments))
        score = np.mean(cross_val_score(clf, X, Y, cv=5))
        if np.isnan(score):
            score = 0

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

    return gp_datas
