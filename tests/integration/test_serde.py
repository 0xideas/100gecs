import numpy as np
import pytest

from gecs.gec import GEC
from gecs.ger import GER


def test_serde(gec, gec_is_serialised, gec_serialisation_path, X, y_class):

    gec2 = GEC.deserialize(gec_serialisation_path, X, y_class)

    for k, v in gec._get_representation().items():
        v2 = getattr(gec2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"


def test_serde(ger, ger_is_serialised, gec_serialisation_path, X, y_real):

    gec2 = GER.deserialize(gec_serialisation_path, X, y_real)

    for k, v in ger._get_representation().items():
        v2 = getattr(gec2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"
