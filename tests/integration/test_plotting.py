import os

import pytest

from gecs.lightgec import LightGEC
from gecs.catger import CatGER
from gecs.catgec import CatGEC
from gecs.lightger import LightGER


@pytest.fixture
def plot_path():
    return "tests/data/outputs/plots/gec"


def test_plotting_gec(plot_path, gec_is_serialised, gec_serialisation_path, X, y_class):
    gec = LightGEC.deserialize(gec_serialisation_path, X, y_class)

    gec.save_plots(plot_path)


def test_plotting_ger(plot_path, ger_is_serialised, ger_serialisation_path, X, y_real):
    ger = LightGER.deserialize(ger_serialisation_path, X, y_real)

    ger.save_plots(plot_path)


def test_plotting_catgec(
    plot_path, catgec_is_serialised, catgec_serialisation_path, X, y_class
):
    catgec = CatGEC.deserialize(catgec_serialisation_path, X, y_class)

    catgec.save_plots(plot_path)


def test_plotting_catger(
    plot_path, catger_is_serialised, catger_serialisation_path, X, y_class
):
    catgec = CatGER.deserialize(catger_serialisation_path, X, y_class)

    catgec.save_plots(plot_path)
