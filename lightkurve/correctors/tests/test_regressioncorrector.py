import numpy as np
from numpy.testing import assert_almost_equal

from ... import LightCurve
from .. import RegressionCorrector, DesignMatrix


def test_sinusoid_noise():
    """Can we remove simple sinusoid noise added to a flat light curve?"""
    size = 100
    time = np.linspace(1, 100, size)
    true_flux = np.ones(size)
    noise = np.sin(time/5)
    # True light curve is flat, i.e. flux=1 at all time steps
    true_lc = LightCurve(time, true_flux, flux_err=0.1*np.ones(size))
    # Noisy light curve has a sinusoid single added
    noisy_lc = LightCurve(time, true_flux+noise, flux_err=true_lc.flux_err)
    dm = DesignMatrix({'noise': noise,
                       'offset': np.ones(len(time))},
                      name='noise_model')

    # Can we recover the true light curve?
    rc = RegressionCorrector(noisy_lc)
    corrected_lc = rc.correct(dm)
    assert_almost_equal(corrected_lc.normalize().flux, true_lc.flux)

    # Can we produce the diagnostic plot?
    rc.diagnose()

    # Does it work when we set priors?
    dm.prior_mu = [0.1, 0.1]
    dm.prior_sigma = [1e6, 1e6]
    corrected_lc = RegressionCorrector(noisy_lc).correct(dm)
    assert_almost_equal(corrected_lc.normalize().flux, true_lc.flux)

    # Does it work when `flux_err` isn't available?
    noisy_lc = LightCurve(time=time, flux=true_flux+noise)
    corrected_lc = RegressionCorrector(noisy_lc).correct(dm)
    assert_almost_equal(corrected_lc.normalize().flux, true_lc.flux)
