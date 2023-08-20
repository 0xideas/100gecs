import os

import pytest

from gecs.gec import GEC
from gecs.gecar import GECar
from gecs.gecat import GECat
from gecs.ger import GER


@pytest.fixture
def plot_path():
    return "tests/data/outputs/plots/gec"


def test_plotting_gec(plot_path, gec_is_serialised, gec_serialisation_path, X, y_class):
    gec = GEC.deserialize(gec_serialisation_path, X, y_class)

    gec.save_plots(plot_path)


def test_plotting_ger(plot_path, ger_is_serialised, ger_serialisation_path, X, y_real):
    ger = GER.deserialize(ger_serialisation_path, X, y_real)

    ger.save_plots(plot_path)


def test_plotting_gecat(plot_path, gecat_is_serialised, gecat_serialisation_path, X, y_class):
    gecat = GECat.deserialize(gecat_serialisation_path, X, y_class)

    gecat.save_plots(plot_path)


def test_plotting_gecar(plot_path, gecar_is_serialised, gecar_serialisation_path, X, y_class):
    gecat = GECar.deserialize(gecar_serialisation_path, X, y_class)

    gecat.save_plots(plot_path)