import pytest

from astropy import units as u
import numpy as np
from numpy.testing import assert_array_equal

from ..periodogram import Periodogram
from ..lightcurvefile import KeplerLightCurveFile

# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")


@pytest.mark.remote_data
def test_from_lightcurve():
    """Can we create a Periodogram using `from_lightcurve`?"""
    lc = KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX
    Periodogram.from_lightcurve(lc=lc)
    # Can we provide frequencies as a list?
    frequencies = [0.1, 0.2, 0.3]
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies.value, frequencies)
    assert(pg.frequencies.unit == u.microhertz)
    # Can we provide frequencies as a Quantity in uHz?
    frequencies = np.array(frequencies) * u.microhertz
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies, frequencies)
    # Can we provide frequencies as a Quantity in Hz?
    frequencies = np.array(frequencies) * u.hertz
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies, frequencies)


@pytest.mark.remote_data
def test_lightcurve_seismology_plot():
    """Sanity check to verify that periodogram plotting works"""
    KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX.periodogram().plot()
