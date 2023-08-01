import numpy as np
import pytest

from gecs.gec import GEC


def test_serde(gec, gec_is_serialised, serialisation_path, X, y):

    gec2 = GEC.deserialize(serialisation_path, X, y)

    for k, v in gec._get_representation().items():
        v2 = getattr(gec2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"
