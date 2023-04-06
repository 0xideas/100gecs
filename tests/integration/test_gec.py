import pytest
import numpy as np
from gecs100.gec import GEC


@pytest.fixture(scope="session")
def gec():
    return GEC()


@pytest.fixture(scope="session")
def seed():
    return 102


@pytest.fixture(scope="session")
def X(seed):
    np.random.seed(seed)
    return np.random.randn(100, 3)


@pytest.fixture(scope="session")
def y(X, seed):
    np.random.seed(seed)
    return np.array(
        [
            min(4, yy)
            for yy in ((X.sum(1) ** 2 + np.random.uniform(0, 1, X.shape[0]))).astype(
                int
            )
        ]
    )


def test_gec(gec, X, y):
    gec.fit(X, y, 1)
    y_hat = gec.predict(X)

    assert y.shape == y_hat.shape
    assert np.mean(y == y_hat) > 0.0
