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


def test_sine_SFF():
    """Test the SFF implementation with known signals"""
    # Retrieve the custom, known signal properties
    tpf_sine = KeplerTargetPixelFile(filename_synthetic_sine)
    true_period = np.float(tpf_sine.hdu[3].header['PERIOD'])
    true_amplitude = np.float(tpf_sine.hdu[3].header['SINE_AMP'])

    # Run the SFF algorithm
    lc_sine = tpf_sine.to_lightcurve()
    corrector = SFFCorrector(lc_sine)
    cor_sine = corrector.correct(tpf_sine.pos_corr2, tpf_sine.pos_corr1,
                                 niters=4, windows=5, bins=7, restore_trend=True)

    # Verify that we get the period within ~20%
    pg = cor_sine.to_periodogram(method='lombscargle', minimum_period=1,
                                 maximum_period=10, oversample_factor=10)
    ret_period = pg.period_at_max_power.value
    threshold = 0.2
    assert ((ret_period > true_period*(1-threshold)) &
            (ret_period < true_period*(1+threshold)) )

    # Verify that we get the amplitude to within a factor of 2
    n_cad = len(tpf_sine.time)
    design_matrix = np.vstack([np.ones(n_cad),
                              np.sin(2.0*np.pi*cor_sine.time/ret_period),
                              np.cos(2.0*np.pi*cor_sine.time/ret_period)]).T
    ATA = np.dot(design_matrix.T,  design_matrix / cor_sine.flux_err[:, None]**2)
    least_squares_coeffs = np.linalg.solve(ATA, np.dot(design_matrix.T, cor_sine.flux/cor_sine.flux_err**2 ))
    const, sin_weight, cos_weight = least_squares_coeffs

    fractional_amplitude = (sin_weight**2+cos_weight**2)**(0.5) / const
    assert ((fractional_amplitude > true_amplitude/2) &
            (fractional_amplitude < true_amplitude*2) )
