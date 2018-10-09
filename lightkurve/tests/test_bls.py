from __future__ import division, print_function
import pytest
from bls import BLS
import numpy as np
from ..lightcurvefile import KeplerLightCurveFile


KEPLER10 = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
            "0119/011904151/kplr011904151-2010009091648_llc.fits")


@pytest.mark.remote_data
def test_kepler_10():
    """Make sure you can recover kepler 10"""
    lcf = KeplerLightCurveFile(KEPLER10)
    # Clip out the outliers
    lc = lcf.PDCSAP_FLUX
    temp_lc = lc.remove_outliers(10).flatten().remove_outliers(6)
    lc =  lc[np.in1d(lcf.time, temp_lc.time)].flatten()
    # Find the BLS power spectrum
    model = BLS(lc.time, lc.flux).power(np.linspace(0.7, 0.9, 1000), np.linspace(0.001, 0.15, 10))
    assert np.isclose(model.period[np.argmax(model.power)], 0.837537, rtol=1e-4)
