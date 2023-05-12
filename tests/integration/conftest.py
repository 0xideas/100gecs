import pytest
import numpy as np
from gecs100.gec import GEC


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


@pytest.fixture(scope="session")
def gec(X, y):
    gec = GEC()
    gec.fit(X, y, 3)
    return gec
