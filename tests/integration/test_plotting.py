import pytest
from gecs100.gec import GEC

@pytest.fixture
def plot_path():
    return "tests/data/plots/gec"


def test_plotting(plot_path):
    gec = GEC.deserialise("tests/data/gec_for_plot_test.json")

    gec.save_plots(plot_path)
