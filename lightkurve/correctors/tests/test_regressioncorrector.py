import numpy as np
from numpy.testing import assert_almost_equal

from ... import LightCurve
from .. import RegressionCorrector, DesignMatrix


def test_sinusoid_noise():
    """Can we remove simple sinusoid noise added to a flat light curve?"""
    size = 100
    time = np.linspace(1, 100, size)
    noise = np.sin(time/5)
    true_lc = LightCurve(time, np.ones(size), flux_err=0.1*np.ones(size))
    noisy_lc = LightCurve(time, true_lc.flux+noise, true_lc.flux_err)
    dm = DesignMatrix({'noise': noise, 'offset': np.ones(len(time))},
                      name='noise_model')
    # Ridge with alpha=0 should equal ordinary least squares
    rc = RegressionCorrector(noisy_lc)
    corrected_lc = rc.correct(dm)
    assert_almost_equal(corrected_lc.normalize().flux, true_lc.flux)

    # Can plot
    rc.diagnose()
