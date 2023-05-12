import pytest
import numpy as np
from gecs100.gec import GEC


@pytest.fixture
def serialisation_path():
    return "tests/data/gec.json"


def test_serde(gec, serialisation_path):
    gec.serialise(serialisation_path)

    gec2 = GEC.deserialise(serialisation_path)

    for k, v in gec._get_representation().items():
        if not str(v) == str(getattr(gec2, k)):
            str1 = np.array(list(str(v)))
            str2 = np.array(list(str(getattr(gec2, k))))
            comp = str1 != str2
            print(k)
            print(comp.shape)
            print(np.sum(comp))
            print(list(comp.astype(int)))
            print(str1[comp])
            print(str2[comp])

            assert False
