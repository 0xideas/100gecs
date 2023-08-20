import numpy as np
import pytest

from gecs.lightgec import LightGEC
from gecs.catger import CatGER
from gecs.catgec import CatGEC
from gecs.lightger import LightGER


def test_serde_gec(gec, gec_is_serialised, gec_serialisation_path, X, y_class):

    gec2 = GEC.deserialize(gec_serialisation_path, X, y_class)

    for k, v in gec._get_representation().items():
        v2 = getattr(gec2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"


def test_serde_ger(ger, ger_is_serialised, ger_serialisation_path, X, y_real):

    ger2 = GER.deserialize(ger_serialisation_path, X, y_real)

    for k, v in ger._get_representation().items():
        v2 = getattr(ger2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"


def test_serde_catgec(catgec, catgec_is_serialised, catgec_serialisation_path, X, y_class):

    catgec = CatGEC.deserialize(catgec_serialisation_path, X, y_class)

    for k, v in catgec._get_representation().items():
        v2 = getattr(catgec, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"


def test_serde_catger(catger, catger_is_serialised, catger_serialisation_path, X, y_class):

    catger = CatGER.deserialize(catger_serialisation_path, X, y_class)

    for k, v in catger._get_representation().items():
        v2 = getattr(catger, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"
