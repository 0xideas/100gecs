import numpy as np


def test_gec(gec, X, y_class):
    y_hat = gec.predict(X)

    assert y_class.shape == y_hat.shape
    assert np.mean(y_class == y_hat) > 0.0


def test_ger(ger, X, y_real):
    y_hat = ger.predict(X)

    assert y_real.shape == y_hat.shape
    assert np.corrcoef(y_real, y_hat)[0, 1] > 0.0, f"{y_real}, {y_hat}"


def test_catgec(catgec, X, y_class):
    y_hat = catgec.predict(X)

    assert y_class.shape == y_hat.flatten().shape
    assert np.mean(y_class == y_hat) > 0.0


def test_catger(catger, X, y_real):
    y_hat = catger.predict(X)

    assert y_real.shape == y_hat.flatten().shape
    assert np.corrcoef(y_real, y_hat)[0, 1] > 0.0, f"{y_real}, {y_hat}"
