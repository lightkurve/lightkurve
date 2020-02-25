import pytest

from astropy.utils.data import get_pkg_data_filename

from ... import open as lkopen
from .. import TessPLDCorrector


TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000"
            "/004/176/tess2019128220341-0000000417699452-0016-s_tp.fits")


@pytest.mark.remote_data
def test_basics():
    tpf = lkopen(TESS_SIM)
    corrector = TessPLDCorrector(tpf)
    corrector.correct(spline_n_knots=3, spline_degree=2, pixel_components=1)
    corrector.diagnose()
