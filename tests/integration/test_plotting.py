import os

import pytest

from gecs.gec import GEC


@pytest.fixture
def plot_path():
    return "tests/data/outputs/plots/gec"


@pytest.mark.dependency(depends=["test_serde"])
def test_plotting(plot_path, gec_is_serialised, serialisation_path, X, y):
    gec = GEC.deserialize(serialisation_path, X, y)

    gec.save_plots(plot_path)
