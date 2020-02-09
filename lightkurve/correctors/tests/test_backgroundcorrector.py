import pytest

from astropy.utils.data import get_pkg_data_filename

from ... import open as lkopen
from .. import BackgroundCorrector


TESS_TPF = get_pkg_data_filename("../../tests/data/tess25155310-s01-first-cadences.fits.gz")


def test_basics():
    tpf = lkopen(TESS_TPF)
    corrector = BackgroundCorrector(tpf)
    corrector.correct(spline_n_knots=3)
