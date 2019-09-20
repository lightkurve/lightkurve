import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from ... import LightCurve
from .. import RegressionCorrector, DesignMatrix


def test_regression_corrector():
    size = 50
    time = np.linspace(1, 100, size)
    flux = np.ones(size) + 2*time
    flux = np.ones(cadences) + np.ones(cadences)
    np.sin
    dm = DesignMatrix({})
    RegressionCorrector()


def test_simple_example():
    """Can we remove simple sinusoid noise added to a flat light curve?"""
    size = 100
    time = np.linspace(1, 100, size)
    true_flux = np.ones(size)
    flux_err = 0.1*np.ones(size)
    noise = np.sin(time/5)
    lc = LightCurve(time, true_flux+noise, flux_err)

    dm = DesignMatrix({'noise': noise, 'offset': np.ones(len(noise))})
    rc = RegressionCorrector(lc, dm)

    corrected_lc = rc.correct()
    assert_almost_equal(corrected_lc.normalize().flux, true_flux)
