import numpy as np
import pytest
from numpy.testing import (assert_allclose)

from ... import LightCurve, search_lightcurve
from ..metrics import overfit_metric_lombscargle, underfit_metric_neighbors, _compute_correlation


def test_overfit_metric_lombscargle():
    """Sanity checks for `overfit_metric_lombscargle`"""
    # Create artificial flat and sinusoid light curves
    time = np.arange(1, 100, 0.1)
    lc_flat = LightCurve(time=time, flux=1, flux_err=0.0)
    lc_sine = LightCurve(time=time, flux=np.sin(time) + 1, flux_err=0.0)

    # If the light curve didn't change, it should be "perfect", i.e. metric == 1
    assert overfit_metric_lombscargle(lc_flat, lc_flat) == 1.0
    assert overfit_metric_lombscargle(lc_sine, lc_sine) == 1.0

    # If the light curve went from a sine to a flat line,
    # no noise was introduced, hence metric == 1 (good)
    assert overfit_metric_lombscargle(lc_sine, lc_flat) == 1.0

    # If the light curve went from flat to sine, metric == 0 (bad)
    assert overfit_metric_lombscargle(lc_flat, lc_sine) == 0.0
    # However, if the light curves were noisy to begin with, it shouldn't be considered that bad
    lc_flat.flux_err += 0.5
    lc_sine.flux_err += 0.5
    assert overfit_metric_lombscargle(lc_flat, lc_sine) > 0.5


@pytest.mark.remote_data
def test_underfit_metric_neighbors():
    """Sanity checks for `underfit_metric_neighbors`."""
    # PDCSAP_FLUX has a very good score (>0.99) because it has been corrected
    lc_pdcsap = search_lightcurve("Proxima Cen", sector=11, author="SPOC").download(
        flux_column="pdcsap_flux"
    )
    assert underfit_metric_neighbors(lc_pdcsap, min_targets=3, max_targets=3) > 0.99

    # SAP_FLUX has a worse score (<0.9) because it hasn't been corrected
    lc_sap = lc_pdcsap.copy()
    lc_sap.flux = lc_pdcsap.sap_flux
    lc_sap.flux_err = lc_pdcsap.sap_flux_err
    assert underfit_metric_neighbors(lc_sap, min_targets=3, max_targets=3) < 0.9

    # A flat light curve should have a perfect score (1)
    notnan = ~np.isnan(lc_sap.flux)
    lc_sap.flux.value[notnan] = np.ones(notnan.sum())
    assert underfit_metric_neighbors(lc_sap, min_targets=3, max_targets=3) == 1.0



def test_compute_correlation():
    """ Simple test to verify the correction function works"""
    
    # Fully correlated matrix
    fluxMatrix = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
    correlation_matrix = _compute_correlation(fluxMatrix)
    assert np.all(correlation_matrix == 1.0)

    # Partially correlated
    fluxMatrix = np.array([[ 1.0, -1.0,  1.0, -1.0], 
                           [-1.0,  1.0,  1.0, -1.0], 
                           [ 1.0, -1.0,  1.0, -1.0], 
                           [-1.0,  1.0, -1.0,  1.0]])
    correlation_matrix = _compute_correlation(fluxMatrix)
    correlation_truth = np.array([[ 1.0, -1.0,  0.5, -0.5], 
                           [-1.0,  1.0, -0.5,  0.5], 
                           [ 0.5, -0.5,  1.0, -1.0], 
                           [-0.5,  0.5, -1.0,  1.0]])
    assert_allclose(correlation_matrix, correlation_truth)
