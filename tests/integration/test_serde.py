import numpy as np
import pytest

from gecs.gec import GEC
from gecs.gecar import GECar
from gecs.gecat import GECat
from gecs.ger import GER


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



def test_serde_gecat(gecat, gecat_is_serialised, gecat_serialisation_path, X, y_class):

    gecat = GECat.deserialize(gecat_serialisation_path, X, y_class)

    for k, v in gecat._get_representation().items():
        v2 = getattr(gecat, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"



def test_serde_gecar(gecar, gecar_is_serialised, gecar_serialisation_path, X, y_class):

    gecar = GECar.deserialize(gecar_serialisation_path, X, y_class)

    for k, v in gecar._get_representation().items():
        v2 = getattr(gecar, f"{k}_")
        assert str(v) == str(v2), f"{k} - {v} != {v2}"