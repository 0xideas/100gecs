# monkeytype run monkeytyping/run_gers.py
# monkeytype apply gecs.gec_base
# monkeytype apply gecs.catger
# monkeytype apply gecs.lightger


import os

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score

from gecs.catger import CatGER
from gecs.lightger import LightGER

if __name__ == "__main__":
    gers_ = [(LightGER, LightGER()), (CatGER, CatGER())]

    X, y = load_diabetes(return_X_y=True)

    # fit and infer GEC
    for ger_class, ger in gers_:
        ger.fit(X, y)
        _ = ger.predict(X)

        path = "./ger.json"
        ger.serialize(path)
        _ = ger_class.deserialize(path, X, y)
        _ = ger_class.deserialize(path)
        os.remove(path)

        ger.freeze()
        ger.fit(X, y)
        _ = ger.predict(X)
        ger_score = np.mean(cross_val_score(ger, X, y))
        ger.unfreeze()
