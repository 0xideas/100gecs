import pytest
import numpy as np
from gecs100.gec import GEC


def test_gec(gec, X, y):
    y_hat = gec.predict(X)

    assert y.shape == y_hat.shape
    assert np.mean(y == y_hat) > 0.0
