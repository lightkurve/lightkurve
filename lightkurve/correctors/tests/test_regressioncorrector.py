import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename


from ..regressioncorrector import RegressionCorrector
from ...lightcurve import KeplerLightCurve, LightCurve
from ...utils import LightkurveError
from ...search import open

filename_test = get_pkg_data_filename("data/test_lc_k2-18.fits")


def test_init():
    # Needs centroids
    lc = LightCurve(np.arange(10), np.ones(10))
    err_string = ("Input light curve does not have a centroid_col attribute.")
    with pytest.raises(LightkurveError) as err:
        r = RegressionCorrector(lc)
    assert err_string in err.value.args[0]

    lc = KeplerLightCurve(np.arange(10), np.ones(10))
    err_string = ("Input light curve centroid_col attribute is all nans.")
    with pytest.raises(LightkurveError) as err:
        r = RegressionCorrector(lc)
    assert err_string in err.value.args[0]
    lc = KeplerLightCurve(np.arange(10), np.ones(10), centroid_col=np.ones(10), centroid_row=np.ones(10))
    r = RegressionCorrector(lc)

def test_correct():
    lc = open(filename_test).get_lightcurve('FLUX')
    r = RegressionCorrector(lc)

    r.correct()

    design_matrix = np.vstack([lc.centroid_col, lc.centroid_row]).T
    r.correct(design_matrix)
