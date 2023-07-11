import numpy as np
import pytest

from gecs.gec import GEC


@pytest.fixture
def serialisation_path():
    return "tests/data/outputs/gec.json"


def test_serde(gec, serialisation_path, X, y):
    gec.serialize(serialisation_path)

    gec2 = GEC.deserialize(serialisation_path, X, y)

    for k, v in gec._get_representation().items():
        assert str(v) == str(getattr(gec2, k))
