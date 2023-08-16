import numpy as np


def test_gec(gec, X, y_class):
    y_hat = gec.predict(X)

    assert y_class.shape == y_hat.shape
    assert np.mean(y_class == y_hat) > 0.0


def test_ger(ger, X, y_real):
    y_hat = ger.predict(X)

    assert y_real.shape == y_hat.shape
    assert np.corrcoef(y_real, y_hat)[0, 1] > 0.0
