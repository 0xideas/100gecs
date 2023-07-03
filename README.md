![a gecko looking at the camera with bayesian math in white on a pink and green background](documentation/assets/header.png)


# (100)gecs

Bayesian hyperparameter tuning for LGBMClassifier with a scikit-learn API

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The package `gecs` provides the class `GEC`, which is a child class of the class `LGBMClassifier` from the package `lightgbm`. It can be imported from `gecs.gec` and then used in place of `LGBMClassifier`, with the same API. The difference to `LGBMClassifier` lies in the fact that `GEC`automatically does bayesian hyperparameter optimization of the parameters `learning_rate`, `reg_alpha`, `reg_lambda`, `min_child_samples`, `min_child_weight`, `colsample_bytree` and optionally also of `num_leaves` and `n_estimators`.

The fit method has two new parameters: `n_iter`, which sets the number of hyperparameter combinations that will be tried (the higher `n_iter` the higher the expected accuracy, but at a cost of compute) and `fixed_hyperparameters`, which determines which hyperparameters of the LGBM classifier won't get optimized. By default, these are `n_estimators` and `num_leaves`, as the highest possible value for these hyperparameters is almost always optimal. The idea then is to set these as high as makes sense in a specific context and then optimize the other hyperparameters.


## Installation

    pip install gecs

## Usage

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