import pytest
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose)
from astropy.io import fits as pyfits
from ..periodogram import Periodogram
from ..lightcurve import (LightCurve, KeplerLightCurve, TessLightCurve,
                          iterative_box_period_search)
from ..lightcurvefile import KeplerLightCurveFile, TessLightCurveFile

# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")

@pytest.mark.remote_data
def test_lightcurve_seismology_plot():
    """Sanity check to verify that periodogram plotting works"""
    lcf = KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX.normalize()
    pf = Periodogram.from_lightcurve(lcf)
    pf.plot()

    lcf.Periodogram().plot()
