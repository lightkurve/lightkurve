"""Unit tests for the `RegressionCorrector` class."""
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
import pytest

from ... import LightCurve, LightkurveWarning
from .. import RegressionCorrector, DesignMatrix


def test_regressioncorrector_priors():
    """This test will fit a design matrix containing the column vectors
    a=[1, 1] and b=[1, 2] to a light curve with flux=[5, 10].

    The best coefficients for this problem are [0, 5] because 0*a + 5*b == flux,
    however we will verify that changing the priors will yield different
    solutions.
    """
    lc1 = LightCurve(flux=[5, 10])
    lc2 = LightCurve(flux=[5, 10], flux_err=[1, 1])
    design_matrix = DesignMatrix(pd.DataFrame({'a':[1, 1], 'b':[1, 2]}))
    for dm in [design_matrix, design_matrix.to_sparse()]:
        for lc in [lc1, lc2]:
            rc = RegressionCorrector(lc)

            # No prior
            rc.correct(dm)
            assert_almost_equal(rc.coefficients, [0, 5])

            # Strict prior centered on correct solution
            dm.prior_mu = [0, 5]
            dm.prior_sigma = [1e-6, 1e-6]
            rc.correct(dm)
            assert_almost_equal(rc.coefficients, [0, 5])

            # Strict prior centered on incorrect solution
            dm.prior_mu = [99, 99]
            dm.prior_sigma = [1e-6, 1e-6]
            rc.correct(dm)
            assert_almost_equal(rc.coefficients, [99, 99])

            # Wide prior centered on incorrect solution
            dm.prior_mu = [9, 9]
            dm.prior_sigma = [1e6, 1e6]
            rc.correct(dm)
            assert_almost_equal(rc.coefficients, [0, 5])

def test_sinusoid_noise():
    """Can we remove simple sinusoid noise added to a flat light curve?"""
    size = 100
    time = np.linspace(1, 100, size)
    true_flux = np.ones(size)
    noise = np.sin(time/5)
    # True light curve is flat, i.e. flux=1 at all time steps
    true_lc = LightCurve(time=time, flux=true_flux, flux_err=0.1*np.ones(size))
    # Noisy light curve has a sinusoid single added
    noisy_lc = LightCurve(time=time, flux=true_flux+noise, flux_err=true_lc.flux_err)
    design_matrix = DesignMatrix({'noise': noise,
                                  'offset': np.ones(len(time))},
                                  name='noise_model')

    for dm in [design_matrix, design_matrix.to_sparse()]:
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


def test_nan_input():
    # The following light curves should all raise ValueErrors because of NaNs
    with warnings.catch_warnings():
        # Instantiating light curves with NaN times will yield a warning
        warnings.simplefilter("ignore", LightkurveWarning)
        lcs = [LightCurve(flux=[5, 10], flux_err=[np.nan, 1]),
               LightCurve(flux=[np.nan, 10], flux_err=[1, 1])]

    # Passing these to RegressionCorrector should raise a ValueError
    for lc in lcs:
        with pytest.raises(ValueError):
            RegressionCorrector(lc)

    # However, we should be flexible with letting `flux_err` be all-NaNs,
    # because it is common for errors to be missing.
    lc = LightCurve(flux=[5, 10], flux_err=[np.nan, np.nan])
    RegressionCorrector(lc)


def test_zero_fluxerr():
    """Regression test for #668.

    Flux uncertainties smaller than or equal to zero (`lc.flux_err <= 0`) will
    trigger an invalid or non-finite matrix.  We expect `RegressionCorrector`
    to detect this and yield a graceful `ValueError`."""
    lc = LightCurve(flux=[5, 10], flux_err=[1, 0])
    with pytest.raises(ValueError):
        RegressionCorrector(lc)
    lc = LightCurve(flux=[5, 10], flux_err=[1, -10])
    with pytest.raises(ValueError):
        RegressionCorrector(lc)
