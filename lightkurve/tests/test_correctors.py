from __future__ import division, print_function

import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from astropy.utils.data import get_pkg_data_filename

from ..lightcurve import KeplerLightCurve, LightCurve
from ..lightcurvefile import KeplerLightCurveFile
from ..correctors import KeplerCBVCorrector, SFFCorrector, GPCorrector

from .test_lightcurve import TABBY_Q8


def test_basic_GPcorrector():
    # Make a short, fake light curve
    x = np.arange(0, 10, 0.01)
    y = np.ones(len(x))
    y = np.sin(5 * (x/np.pi))
    y += np.random.normal(0, 0.1, len(y))
    y += 10
    y[400:410] = np.nan
    y_err = np.zeros(len(y)) + 0.1
    lc = LightCurve(x, y, flux_err=y_err)

    with pytest.raises(ValueError) as exc:
        GPCorrector(lc).correct()
    assert('Light curve contains NaN values.' in str(exc))

    with pytest.raises(ValueError) as exc:
        GPCorrector(lc.remove_nans()).correct()
    assert('Flux is unnormalized.' in str(exc))
    lc = lc.remove_nans().normalize()
    corr = GPCorrector(lc).correct()
    assert corr.flux.shape == lc.flux.shape
    assert np.nansum(corr.flux) != np.nansum(lc.flux)
    # Standard deviation should go down
    assert corr.flux.std() < lc.flux.std()
    # But all the errors should increase
    assert np.all(corr.flux_err - lc.flux_err > 0)


@pytest.mark.remote_data
def test_GPcorrector():
    lc = KeplerLightCurveFile.from_archive(
        'Kepler-102', quarter=5).PDCSAP_FLUX.remove_nans().normalize()
    gp = GPCorrector(lc)
    corr, trend = gp.correct(iters=5, sigma=3, return_trend=True)
    flat = lc.flatten()
    assert lc.flux.shape == corr.flux.shape
    assert corr.flux.shape == trend.flux.shape
    # Corr should be detrended
    assert corr.flux.std() < trend.flux.std()
    # Corrected should have larger errors than trend
    assert np.all((corr.flux_err - trend.flux_err > 0))
    # Should have a mask object
    assert gp.mask.shape == lc.flux.shape
    # Should have a mask object
    assert not np.all(gp.mask)
    # Should beat out savgol for a transit case
    assert corr.fold(16.14570).bin(10).flux.min() < flat.fold(16.14570).bin(10).flux.min()


@pytest.mark.remote_data
def test_kepler_cbv_fit():
    # comparing that the two methods to do cbv fit are the nearly the same
    cbv = KeplerCBVCorrector(TABBY_Q8)
    cbv_lc = cbv.correct()
    assert_almost_equal(cbv.coeffs, [0.08534423, 0.10814261], decimal=3)
    lcf = KeplerLightCurveFile(TABBY_Q8)
    cbv_lcf = lcf.compute_cotrended_lightcurve()
    assert_almost_equal(cbv_lc.flux, cbv_lcf.flux)


def test_sff_corrector():
    """Does our code agree with the example presented in Vanderburg
    and Jhonson (2014)?"""
    # The following csv file, provided by Vanderburg and Jhonson
    # at https://www.cfa.harvard.edu/~avanderb/k2/ep60021426.html,
    # contains the results of applying SFF to EPIC 60021426.
    fn = get_pkg_data_filename('./data/ep60021426alldiagnostics.csv')
    data = np.genfromtxt(fn, delimiter=',', skip_header=1)
    mask = data[:, -2] == 0  # indicates whether the thrusters were on or off
    time = data[:, 0]
    raw_flux = data[:, 1]
    corrected_flux = data[:, 2]
    centroid_col = data[:, 3]
    centroid_row = data[:, 4]
    arclength = data[:, 5]
    correction = data[:, 6]

    sff = SFFCorrector()
    corrected_lc = sff.correct(time=time, flux=raw_flux,
                               centroid_col=centroid_col,
                               centroid_row=centroid_row,
                               niters=1)
    # do hidden plots execute smoothly?
    ax = sff._plot_rotated_centroids()
    ax = sff._plot_normflux_arclength()

    # the factor self.bspline(time-time[0]) accounts for
    # the long term trend which is divided out in order to get a "flat"
    # lightcurve.
    assert_almost_equal(corrected_lc.flux*sff.bspline(time),
                        corrected_flux, decimal=3)
    assert_array_equal(time, corrected_lc.time)
    # the factor of 4 below accounts for the conversion
    # between pixel units to arcseconds
    assert_almost_equal(4*sff.s, arclength, decimal=2)
    assert_almost_equal(sff.interp(sff.s), correction, decimal=3)

    # test using KeplerLightCurve interface
    klc = KeplerLightCurve(time=time, flux=raw_flux, centroid_col=centroid_col,
                           centroid_row=centroid_row)
    klc = klc.correct(niters=1)
    sff = klc.corrector

    assert_almost_equal(klc.flux*sff.bspline(time),
                        corrected_flux, decimal=3)
    assert_almost_equal(4*sff.s, arclength, decimal=2)
    assert_almost_equal(sff.interp(sff.s), correction, decimal=3)
    assert_array_equal(time, klc.time)
