"""Use synthetic data to verify lightkurve detrending and signal recovery.
"""
from __future__ import division, print_function

from astropy.utils.data import get_pkg_data_filename
from astropy.stats.bls import BoxLeastSquares

import numpy as np
import pytest
from scipy import stats

from ..targetpixelfile import KeplerTargetPixelFile
from ..correctors import SFFCorrector, PLDCorrector

# See `data/synthetic/README.md` for details about these synthetic test files
filename_synthetic_sine = get_pkg_data_filename("data/synthetic/synthetic-k2-sinusoid.targ.fits.gz")
filename_synthetic_transit = get_pkg_data_filename("data/synthetic/synthetic-k2-planet.targ.fits.gz")
filename_synthetic_flat = get_pkg_data_filename("data/synthetic/synthetic-k2-flat.targ.fits.gz")


def test_sine_sff():
    """Can we recover a synthetic sine curve using SFF and LombScargle?"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_sine)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_amplitude = np.float(tpf.hdu[3].header['SINE_AMP'])

    # Run the SFF algorithm
    lc = tpf.to_lightcurve()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                               niters=4, windows=1, bins=7, restore_trend=True, timescale=0.5)

    # Verify that we get the period within ~20%
    pg = cor_lc.to_periodogram(method='lombscargle', minimum_period=1,
                               maximum_period=10, oversample_factor=10)
    ret_period = pg.period_at_max_power.value
    threshold = 0.2
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the amplitude to within 10%
    n_cad = len(tpf.time)
    design_matrix = np.vstack([np.ones(n_cad),
                              np.sin(2.0*np.pi*cor_lc.time.value/ret_period),
                              np.cos(2.0*np.pi*cor_lc.time.value/ret_period)]).T
    ATA = np.dot(design_matrix.T,  design_matrix / cor_lc.flux_err[:, None]**2)
    least_squares_coeffs = np.linalg.solve(ATA, np.dot(design_matrix.T, cor_lc.flux/cor_lc.flux_err**2 ))
    const, sin_weight, cos_weight = least_squares_coeffs

    fractional_amplitude = (sin_weight**2+cos_weight**2)**(0.5) / const
    assert ((fractional_amplitude > true_amplitude/1.1) &
            (fractional_amplitude < true_amplitude*1.1) )


def test_transit_sff():
    """Can we recover a synthetic exoplanet signal using SFF and BLS?"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_transit)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_rprs = np.float(tpf.hdu[3].header['RPRS'])
    true_transit_lc = tpf.hdu[3].data['NOISELESS_INPUT']
    max_depth = 1-np.min(true_transit_lc)

    # Run the SFF algorithm
    lc = tpf.to_lightcurve().normalize()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                               niters=4, windows=1, bins=7, restore_trend=False, timescale=0.5)

    # Verify that we get the transit period within 5%
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))
    ret_period = pg.period_at_max_power.value
    threshold = 0.05
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)))

    # Verify that we get the transit depth in expected bounds
    assert ((pg.depth_at_max_power >= true_rprs**2) &
            (pg.depth_at_max_power < max_depth))


def test_transit_pld():
    """Can we recover a synthetic exoplanet signal using PLD and BLS?"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_transit)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_rprs = np.float(tpf.hdu[3].header['RPRS'])
    true_transit_lc = tpf.hdu[3].data['NOISELESS_INPUT']
    max_depth = 1-np.min(true_transit_lc)

    # Run the PLD algorithm on a first pass
    corrector = PLDCorrector(tpf)
    cor_lc = corrector.correct()
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))

    # Re-do PLD with the suspected transits masked
    cor_lc = corrector.correct(cadence_mask=~pg.get_transit_mask()).normalize()
    pg = cor_lc.to_periodogram(method='bls', minimum_period=1, maximum_period=9,
                               frequency_factor=0.05, duration=np.arange(0.1, 0.6, 0.1))

    # Verify that we get the period within ~5%
    ret_period = pg.period_at_max_power.value
    threshold = 0.05
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)))

    # Verify that we get the transit depth in expected bounds
    assert ((pg.depth_at_max_power >= true_rprs**2) &
            (pg.depth_at_max_power < max_depth))


def test_sine_pld():
    """Can we recover a synthetic sine wave using PLD and LombScargle?"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_sine)
    true_period = np.float(tpf.hdu[3].header['PERIOD'])
    true_amplitude = np.float(tpf.hdu[3].header['SINE_AMP'])

    # Run the PLD algorithm
    corrector = tpf.to_corrector('pld')
    cor_lc = corrector.correct()

    # Verify that we get the period within ~20%
    pg = cor_lc.to_periodogram(method='lombscargle', minimum_period=1,
                               maximum_period=10, oversample_factor=10)
    ret_period = pg.period_at_max_power.value
    threshold = 0.2
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the amplitude to within 20%
    n_cad = len(tpf.time)
    design_matrix = np.vstack([np.ones(n_cad),
                              np.sin(2.0*np.pi*cor_lc.time.value/ret_period),
                              np.cos(2.0*np.pi*cor_lc.time.value/ret_period)]).T
    ATA = np.dot(design_matrix.T,  design_matrix / cor_lc.flux_err[:, None]**2)
    least_squares_coeffs = np.linalg.solve(ATA, np.dot(design_matrix.T, cor_lc.flux/cor_lc.flux_err**2 ))
    const, sin_weight, cos_weight = least_squares_coeffs

    fractional_amplitude = (sin_weight**2+cos_weight**2)**(0.5) / const
    assert ((fractional_amplitude > true_amplitude/1.1) &
            (fractional_amplitude < true_amplitude*1.1) )


def test_detrending_residuals():
    """Test the detrending residual distributions"""
    # Retrieve the custom, known signal properties
    tpf = KeplerTargetPixelFile(filename_synthetic_flat)

    # Run the SFF algorithm
    lc = tpf.to_lightcurve()
    corrector = SFFCorrector(lc)
    cor_lc = corrector.correct(tpf.pos_corr2, tpf.pos_corr1,
                               niters=10, windows=5, bins=7, restore_trend=True)

    # Verify that we get a significant reduction in RMS
    cdpp_improvement = lc.estimate_cdpp() / cor_lc.estimate_cdpp()
    assert cdpp_improvement > 10.0

    # The residuals should be Gaussian-"ish"
    # Table 4.1 of Ivezic, Connolly, Vanerplas, Gray 2014
    anderson_threshold = 1.57

    resid_n_sigmas = (cor_lc.flux - np.mean(cor_lc.flux))/cor_lc.flux_err
    A_value, _, _ = stats.anderson(resid_n_sigmas)
    assert A_value**2 < anderson_threshold

    n_sigma = np.std(resid_n_sigmas)
    assert n_sigma < 2.0

    corrector = tpf.to_corrector('pld')
    cor_lc = corrector.correct(restore_trend=False)

    cdpp_improvement = lc.estimate_cdpp()/cor_lc.estimate_cdpp()
    assert cdpp_improvement > 10.0

    resid_n_sigmas = (cor_lc.flux - np.mean(cor_lc.flux))/cor_lc.flux_err
    A_value, crit, sig = stats.anderson(resid_n_sigmas)
    assert A_value**2 < anderson_threshold

    n_sigma = np.std(resid_n_sigmas)
    assert n_sigma < 2.0


def test_centroids():
    """Test the estimate centroid method."""
    for fn in (filename_synthetic_sine, filename_synthetic_transit,
               filename_synthetic_flat):
        tpf = KeplerTargetPixelFile(fn)
        xraw, yraw = tpf.estimate_centroids()
        xnorm = xraw - np.median(xraw)
        ynorm = yraw - np.median(yraw)
        xposc = tpf.pos_corr2 - np.median(tpf.pos_corr2)
        yposc = tpf.pos_corr1 - np.median(tpf.pos_corr1)
        rmax = np.max(np.sqrt((xnorm.value-xposc)**2 + (ynorm.value-yposc)**2))
        # The centroids should agree to within a hundredth of a pixel.
        assert rmax < 0.01
