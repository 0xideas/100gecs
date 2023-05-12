import pytest
import numpy as np
from gecs100.gec import GEC


@pytest.fixture
def serialisation_path():
    return "tests/data/gec.json"


def test_serde(gec, serialisation_path, X, y):
    gec.serialise(serialisation_path)

    gec2 = GEC.deserialise(serialisation_path, X, y)

    for k, v in gec._get_representation().items():
        assert str(v) == str(getattr(gec2, k))
