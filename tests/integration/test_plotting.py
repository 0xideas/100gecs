import pytest
import os
from gecs.gec import GEC


@pytest.fixture
def plot_path():
    return "tests/data/outputs/plots/gec"


def test_plotting(plot_path, X, y):
    gec = GEC.deserialise("tests/data/inputs/gec_for_plot_test.json", X, y)

    gec.save_plots(plot_path)
