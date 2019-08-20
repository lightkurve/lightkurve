import pytest
import numpy as np

from ..regressioncorrector import RegressionCorrector
from ...lightcurve import KeplerLightCurve, LightCurve
from ...utils import LightkurveError

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
