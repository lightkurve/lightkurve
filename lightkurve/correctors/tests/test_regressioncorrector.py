import pytest
import numpy as np
from astropy.utils.data import get_pkg_data_filename


from ..regressioncorrector import RegressionCorrector
from ...lightcurve import KeplerLightCurve, LightCurve
from ...utils import LightkurveError
from ...search import open as lkopen

filename_test = get_pkg_data_filename("data/test_lc_k2-18.fits")


def test_init():
    # Needs centroids
    lc = LightCurve(np.arange(10), np.ones(10))
    err_string = ("Input light curve does not have a centroid_col attribute.")
    with pytest.raises(LightkurveError) as err:
        RegressionCorrector(lc)
    assert err_string in err.value.args[0]

    lc = KeplerLightCurve(np.arange(10), np.ones(10))
    err_string = ("Input light curve centroid_col attribute is all nans.")
    with pytest.raises(LightkurveError) as err:
        RegressionCorrector(lc)
    assert err_string in err.value.args[0]

    lc = KeplerLightCurve(np.arange(10), np.ones(10)*np.nan, centroid_col=np.ones(10), centroid_row=np.ones(10))
    err_string = ("Input light curve has NaNs")
    with pytest.raises(LightkurveError) as err:
        RegressionCorrector(lc)
    assert err_string in err.value.args[0]

    lc = KeplerLightCurve(np.arange(10), np.ones(10), centroid_col=np.ones(10), centroid_row=np.ones(10))
    RegressionCorrector(lc)

def test_correct():
    lc = lkopen(filename_test).get_lightcurve('FLUX')
    r = RegressionCorrector(lc)
    r.correct()

    design_matrix = np.vstack([lc.centroid_col, lc.centroid_row])
    with pytest.raises(LightkurveError) as err:
        r.correct(design_matrix)
    assert 'Design matrix must have shape' in err.value.args[0]

    r.correct(design_matrix.T)
    r.correct(design_matrix.T, method='lombscargle')
    cadence_mask = ~lc.remove_outliers(return_mask=True)[1]
    r.correct(design_matrix.T, preserve_trend=True, cadence_mask=cadence_mask)

def test_diagnose():
    lc = lkopen(filename_test).get_lightcurve('FLUX')
    r = RegressionCorrector(lc)
    r.correct()
    r.diagnose()
