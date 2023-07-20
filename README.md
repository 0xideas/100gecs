![a gecko looking at the camera with bayesian math in white on a pink and green background](documentation/assets/header.png)


# (100)gecs

Bayesian hyperparameter tuning for LGBMClassifier with a scikit-learn API


## Table of Contents

- [Project Overview](#project-overview)
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)


## Project Overview

`gecs` is a tool to help automate the process of hyperparameter tuning for LightGBM classifiers, which can potentially save significant time and computational resources in model building and optimization processes. The `GEC` stands for **G**ood **E**nough **C**lassifier, which allows you to focus on other tasks such as feature engineering. If you deploy 100 of them, you get 100GECs.


## Introduction

The primary class in this package is `GEC`, which is derived from `LGBMClassifier`. Like its parent, GEC can be used to build and train gradient boosting models, but with the added feature of **automated bayesian hyperparameter optimization**. It can be imported from `gecs.gec` and then used in place of `LGBMClassifier`, with the same API.

By default, `GEC` optimizes `boosting_type`, `learning_rate`, `reg_alpha`, `reg_lambda`, `min_child_samples`, `min_child_weight`, `colsample_bytree`, `bagging_freq`, `bagging_fraction` and optionally `num_leaves` and `n_estimators`. Which hyperparameters to tune is fully customizable.


## Installation

    pip install gecs


## Usage


The `GEC` class provides the same API to the user as the `LGBMClassifier` class of `lightgbm`, and additionally:

-   the two additional parameters to the fit method:
    - `n_iter`: Defines the number of hyperparameter combinations that the model should try. More iterations could lead to better model performance, but at the expense of computational resources

    - `fixed_hyperparameters`: Allows the user to specify hyperparameters that the GEC should not optimize. By default, these are `n_estimators` and `num_leaves`. Any of the LGBMClassifier init arguments can be fixed, and so can  `bagging_freq` and `bagging_fraction`, but only jointly. This is done by passing the value `bagging`.

-   the methods `serialize` and `deserialize`, which stores the `GEC` state for the hyperparameter optimization process, **but not the fitted** `LGBMClassifier` **parameters**, to a json file. To store the boosted tree model itself, you have to provide your own serialization or use `pickle`

-   the methods `freeze` and `unfreeze` that turn the `GEC` functionally into a `LGBMClassifier` and back


## Example

The default use of `GEC` would look like this:

    from gecs.gec import GEC

    gec.fit(X, y)

    gec.serialize(path) # stores gec data and settings, but not underlying LGBMClassifier attributes

    gec2 = GEC.deserialize(path, X, y) # X and y are necessary to fit the underlying LGBMClassifier

    yhat = gec.predict(X)

    gec.freeze() # freeze GEC so that it behaves like a LGBMClassifier

    gec.unfreeze() # unfreeze to enable GEC hyperparameter optimisation



## Contributing

If you want to contribute, please reach out and I'll design a process around it.

## License

MIT

## Contact Information

You can find my contact information on my website: [https://leonluithlen.eu](https://leonluithlen.eu)