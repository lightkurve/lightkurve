from __future__ import division, print_function

import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from astropy.utils.data import get_pkg_data_filename

from ..lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from ..lightcurvefile import KeplerLightCurveFile
from ..correctors import KeplerCBVCorrector, SFFCorrector, PLDCorrector
from ..search import search_targetpixelfile

from .test_lightcurve import TABBY_Q8


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
    and Johnson (2014)?"""
    # The following csv file, provided by Vanderburg and Johnson
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

    lc = LightCurve(time=time, flux=raw_flux)
    sff = SFFCorrector(lc)
    corrected_lc = sff.correct(centroid_col=centroid_col,
                               centroid_row=centroid_row,
                               niters=1, windows=1)
    # do hidden plots execute smoothly?
    sff._plot_rotated_centroids()
    sff._plot_normflux_arclength()

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
    klc = klc.correct(niters=1, windows=1)
    sff = klc.corrector

    assert_almost_equal(klc.flux*sff.bspline(time),
                        corrected_flux, decimal=3)
    assert_almost_equal(4*sff.s, arclength, decimal=2)
    assert_almost_equal(sff.interp(sff.s), correction, decimal=3)
    assert_array_equal(time, klc.time)


def test_sff_knots():
    """Is SFF robust against gaps in time and irregular time sampling?

    This test creates a random light curve with gaps in time between
    days 20-30 and days 78-80.  In addition, the time sampling rate changes
    in the interval between day 30 and 78.  SFF should fail without error.
    """
    n_points = 300
    time = np.concatenate((np.linspace(0, 20, int(n_points/3)),
                           np.linspace(30, 78, int(n_points/3)),
                           np.linspace(80, 100, int(n_points/3))
                           ))
    lc = KeplerLightCurve(time=time,
                          flux=np.random.normal(1.0, 0.1, n_points),
                          centroid_col=np.random.normal(1.0, 0.1, n_points),
                          centroid_row=np.random.normal(1.0, 0.1, n_points))
    lc.correct()  # should not raise an exception


@pytest.mark.remote_data
@pytest.mark.skipif(('celerite' not in sys.modules) or ('sklearn' not in sys.modules),
                    reason="PLD requires celerite and scikit-learn")
def test_pld_corrector():
    # download tpf data for a target
    k2_target = 247887989
    k2_tpf = search_targetpixelfile(k2_target).download()
    # instantiate PLD corrector object
    pld = PLDCorrector(k2_tpf[:500])
    # produce a PLD-corrected light curve with a default aperture mask
    corrected_lc = pld.correct()
    # ensure the CDPP was reduced by the corrector
    pld_cdpp = corrected_lc.estimate_cdpp()
    raw_cdpp = k2_tpf.to_lightcurve().estimate_cdpp()
    assert(pld_cdpp < raw_cdpp)
    # make sure the returned object is the correct type (`KeplerLightCurve`)
    assert(isinstance(corrected_lc, KeplerLightCurve))
    # try detrending using a threshold mask
    corrected_lc = pld.correct(aperture_mask='threshold')
    # reduce using fewer principle components
    corrected_lc = pld.correct(n_components_first=10, n_components_second=10)
    # try PLD on a TESS observation
    tess_target = 273985862
    tess_tpf = search_targetpixelfile(tess_target, mission='TESS').download()
    # instantiate PLD corrector object
    pld = PLDCorrector(tess_tpf[:500])
    # produce a PLD-corrected light curve with a pipeline aperture mask
    raw_lc = tess_tpf.to_lightcurve(aperture_mask='pipeline')
    corrected_lc = pld.correct(aperture_mask='pipeline', n_components_first=15,
                               n_components_second=15, use_gp=False)
    # the corrected light curve should have higher precision
    assert(corrected_lc.estimate_cdpp() < raw_lc.estimate_cdpp())
    # make sure the returned object is the correct type (`TessLightCurve`)
    assert(isinstance(corrected_lc, TessLightCurve))
