from __future__ import division, print_function

import numpy as np

from astropy.utils.data import get_pkg_data_filename
import pytest
import warnings

from ..lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from ..lightcurvefile import LightCurveFile, KeplerLightCurveFile, TessLightCurveFile
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from ..utils import LightkurveWarning
from ..correctors import SFFCorrector, PLDCorrector

filename_synthetic_sine = get_pkg_data_filename("data/synthetic/synthetic-k2-sinusoid.targ.fits.gz")
filename_synthetic_transit = get_pkg_data_filename("data/synthetic/synthetic-k2-planet.targ.fits.gz")
filename_synthetic_flat = get_pkg_data_filename("data/synthetic/synthetic-k2-flat.targ.fits.gz")


lacks_BLS = False
try:
    from astropy.stats.bls import BoxLeastSquares
except ImportError:
    lacks_BLS = True

def test_sine_SFF():
    """Test the SFF implementation with known signals"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_sine)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_amplitude = np.float(tpf.hdu[3].header['SINE_AMP'])

    # Run the SFF algorithm
    lc = tpf.to_lightcurve()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                                 niters=4, windows=5, bins=7, restore_trend=True)

    # Verify that we get the period within ~20%
    pg = cor_lc.to_periodogram(method='lombscargle', minimum_period=1,
                                 maximum_period=10, oversample_factor=10)
    ret_period = pg.period_at_max_power.value
    threshold = 0.2
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the amplitude to within a factor of 2
    n_cad = len(tpf.time)
    design_matrix = np.vstack([np.ones(n_cad),
                              np.sin(2.0*np.pi*cor_lc.time/ret_period),
                              np.cos(2.0*np.pi*cor_lc.time/ret_period)]).T
    ATA = np.dot(design_matrix.T,  design_matrix / cor_lc.flux_err[:, None]**2)
    least_squares_coeffs = np.linalg.solve(ATA, np.dot(design_matrix.T, cor_lc.flux/cor_lc.flux_err**2 ))
    const, sin_weight, cos_weight = least_squares_coeffs

    fractional_amplitude = (sin_weight**2+cos_weight**2)**(0.5) / const
    assert ((fractional_amplitude > true_amplitude/2) &
            (fractional_amplitude < true_amplitude*2) )

@pytest.mark.skipif(lacks_BLS, reason="Astropy BLS requires Python 3")
def test_transit_SFF():
    """Test the SFF implementation with known signals"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_transit)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_depth = np.float(tpf.hdu[3].header['RPRS'])**2.0

    # Run the SFF algorithm
    lc = tpf.to_lightcurve()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                                 niters=4, windows=5, bins=7, restore_trend=False)

    # Verify that we get the transit period within 5%
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))
    ret_period = pg.period_at_max_power.value
    threshold = 0.05
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the transit depth to within a factor of 50%
    assert ((pg.depth_at_max_power > true_depth/1.5) &
            (pg.depth_at_max_power < true_depth*1.5))


@pytest.mark.skipif(lacks_BLS, reason="Astropy BLS requires Python 3")
def test_transit_PLD():
    """Test the SFF implementation with known signals"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_transit)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_depth = np.float(tpf.hdu[3].header['RPRS'])**2.0

    # Run the PLD algorithm on a first pass
    corrector = PLDCorrector(tpf)
    cor_lc = corrector.correct(use_gp=False)
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))

    # Re-do PLD with the suspected transits masked
    cor_lc = corrector.correct(use_gp=False, cadence_mask=pg.get_transit_mask()).normalize()
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))

    # Verify that we get the period within ~5%
    ret_period = pg.period_at_max_power.value
    threshold = 0.05
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the transit depth to within a factor of 50%

    print(pg.depth_at_max_power, true_depth)
    assert ((pg.depth_at_max_power > true_depth/1.5) &
            (pg.depth_at_max_power < true_depth*1.5))


def test_sine_PLD():
    """Test the SFF implementation with known signals"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_sine)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_amplitude = np.float(tpf.hdu[3].header['SINE_AMP'])

    # Run the SFF algorithm
    corrector = PLDCorrector(tpf)
    cor_lc = corrector.correct(use_gp=False)

    # Verify that we get the period within ~20%
    pg = cor_lc.to_periodogram(method='lombscargle', minimum_period=1,
                                 maximum_period=10, oversample_factor=10)
    ret_period = pg.period_at_max_power.value
    threshold = 0.2
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the amplitude to within a factor of 2
    n_cad = len(tpf.time)
    design_matrix = np.vstack([np.ones(n_cad),
                              np.sin(2.0*np.pi*cor_lc.time/ret_period),
                              np.cos(2.0*np.pi*cor_lc.time/ret_period)]).T
    ATA = np.dot(design_matrix.T,  design_matrix / cor_lc.flux_err[:, None]**2)
    least_squares_coeffs = np.linalg.solve(ATA, np.dot(design_matrix.T, cor_lc.flux/cor_lc.flux_err**2 ))
    const, sin_weight, cos_weight = least_squares_coeffs

    fractional_amplitude = (sin_weight**2+cos_weight**2)**(0.5) / const
    print(ret_period, fractional_amplitude)
    assert ((fractional_amplitude > true_amplitude/2) &
            (fractional_amplitude < true_amplitude*2) )


def test_cdpp_improvement():
    """Test the SFF and PLD CDPP improvement"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_flat)

    # Run the SFF algorithm
    lc = tpf.to_lightcurve()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                                 niters=10, windows=5, bins=7, restore_trend=True)

    # Verify that we get a significant reduction in RMS
    cdpp_improvement = lc.estimate_cdpp()/cor_lc.estimate_cdpp()
    assert cdpp_improvement > 10.0

    corrector = PLDCorrector(tpf)
    cor_lc = corrector.correct(use_gp=False)

    cdpp_improvement = lc.estimate_cdpp()/cor_lc.estimate_cdpp()
    assert cdpp_improvement > 10.0
