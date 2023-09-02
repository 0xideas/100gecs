# monkeytype run monkeytyping/run_gecs.py
# monkeytype apply gecs.gec_base
# monkeytype apply gecs.catgec
# monkeytype apply gecs.lightgec

import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from gecs.catgec import CatGEC
from gecs.lightgec import LightGEC

if __name__ == "__main__":
    gecs_ = [(LightGEC, LightGEC()), (CatGEC, CatGEC())]

    X, y = load_iris(return_X_y=True)

    # fit and infer GEC
    for gec_class, gec in gecs_:
        gec.fit(X, y)
        _ = gec.predict(X)

        path = "./gec.json"
        gec.serialize(path)
        _ = gec_class.deserialize(path, X, y)
        _ = gec_class.deserialize(path)
        os.remove(path)

        gec.freeze()
        gec.fit(X, y)
        _ = gec.predict(X)
        gec_score = np.mean(cross_val_score(gec, X, y))
        gec.unfreeze()
