import numpy as np
import pytest

from gecs.lightgec import LightGEC
from gecs.catger import CatGER
from gecs.catgec import CatGEC
from gecs.lightger import LightGER


def test_serde_gec(lightgec, lightgec_is_serialised, lightgec_serialisation_path, X, y_class):

    lightgec2 =LightGEC.deserialize(lightgec_serialisation_path, X, y_class)

    for k, v in lightgec._get_representation().items():
        v2 = getattr(lightgec2, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"


def test_serde_ger(lightger, lightger_is_serialised, lightger_serialisation_path, X, y_real):

    lightger2 = LightGER.deserialize(lightger_serialisation_path, X, y_real)

    for k, v in lightger._get_representation().items():
        v2 = getattr(lightger2, f"{k}_")
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
