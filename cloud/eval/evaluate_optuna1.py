"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.

In this example, we optimize the validation log loss of cancer detection.

"""

import numpy as np
import optuna.integration.lightgbm as lgb
import sklearn.datasets
from cloud_io.load_dataset import load_dataset
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data, target = load_dataset("bank", "../../data/bank-full.csv")
train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
dtrain = lgb.Dataset(train_x, label=train_y)
dval = lgb.Dataset(val_x, label=val_y)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
}

model = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    callbacks=[early_stopping(50), log_evaluation(50)],
)

prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
accuracy = accuracy_score(val_y, prediction)

best_params = model.params
print("Best params:", best_params)
print("  Accuracy = {}".format(accuracy))
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))
