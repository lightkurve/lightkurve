""" cbvcorrector.py module unit tests
"""

import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose, assert_raises)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd

from ... import TessLightCurve, KeplerLightCurve
from ... import search_lightcurve
from ..designmatrix import DesignMatrix
from ..cbvcorrector import MAX_NUMBER_CBVS
from ..cbvcorrector import get_kepler_cbvs, get_tess_cbvs, \
    CotrendingBasisVectors, KeplerCotrendingBasisVectors, \
    TessCotrendingBasisVectors
from ..cbvcorrector import compute_correlation, CBVCorrector


def test_CotrendingBasisVectors_nonretrieval():
    """ Tests CotrendingBasisVectors class without requiring remote data
    """

    #***
    # Constructor
    # Create some simple CotrendingBasisVectors objects
    # Kepler
    desired_indices = [1,2,4,6,8]
    cbvs = CotrendingBasisVectors('Kepler', cbv_type='SingleScale',
            cbv_indices=desired_indices)
    assert cbvs.mission == 'Kepler'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.cbv_indices == desired_indices
    assert cbvs.band is None

    # TESS
    cbvs = CotrendingBasisVectors('TESS', cbv_type='MultiScale',
            cbv_indices=desired_indices)
    assert cbvs.mission == 'TESS'
    assert cbvs.cbv_type == 'MultiScale'
    assert cbvs.cbv_indices == desired_indices
    assert cbvs.band is None

    # Test using cbv_indices='ALL'
    cbvs = CotrendingBasisVectors('Kepler', cbv_type='SingleScale',
            cbv_indices='ALL')
    assert cbvs.mission == 'Kepler'
    assert cbvs.cbv_type == 'SingleScale'
    assert np.all(cbvs.cbv_indices == np.arange(1,MAX_NUMBER_CBVS+1))
    assert cbvs.band is None

    # These should fail
    with pytest.raises(ValueError):
        CotrendingBasisVectors('SuperKepler', cbv_type='SingleScale',
            cbv_indices=desired_indices)
    with pytest.raises(ValueError):
        CotrendingBasisVectors('Kepler', cbv_type='SuperSingleScale',
            cbv_indices=desired_indices)

    #***
    # _cbvs_to_matrix
    # Make sure CBVs are the columns in the returned 2-dim array
    cbvs = CotrendingBasisVectors('TESS', cbv_type='SingleScale',
            cbv_indices=[1,2,3,4])
    cbvs.cbv_array = np.array([[1.0,2.0,3.0,4.0], [5.0,6.0,7.0,8.0], 
        [9.0,10.0,11.0,12.0], [13.0,14.0,15.0,16.0]])
    cbvMatrix = cbvs._cbvs_to_matrix(cbv_indices=[1,3])
    assert np.all(cbvMatrix == np.array([[1.0,9.0], [2.0,10.0], 
                                [3.0,11.0], [4.0,12.0]]))

    #***
    # plot_cbvs
    cbvs.gap_indicators = np.array([False, True, False, False])
    cbvs.cadenceno = np.array([1,2,3,4])
    cbvs.sector = 1
    cbvs.camera = 2
    cbvs.ccd = 3
    ax = cbvs.plot_cbvs(cbv_indices=[2,4], ax=None)
    assert isinstance(ax, matplotlib.axes.Axes)

    #***
    # align
    # Set up some cadenceno such that both the CBV and the LC are trimmed
    sample_lc = TessLightCurve([1,2,3,4], [1,2,3,4], [0.1, 0.1, 0.1, 0.1],
            cadenceno=[1,3,4,5])
    trimmed_lc = cbvs.align(sample_lc, trim_lc=True)
    assert np.all(trimmed_lc.cadenceno == cbvs.cadenceno)
    assert len(cbvs.cbv_array[0]) == 3
    assert len(trimmed_lc.flux) == 3

@pytest.mark.remote_data
def test_cbv_retrieval():
    """ Tests reading in some CBVs from MAST

        This indirectly tests the classes KeplerCotrendingBasisVectors and
        TessCotrendingBasisVectors

    """
    cbvs = get_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4, cbv_indices=np.arange(1,7))
    assert isinstance(cbvs, KeplerCotrendingBasisVectors)
    ax = cbvs.plot_cbvs('ALL')
    assert isinstance(ax, matplotlib.axes.Axes)

    cbvs = get_kepler_cbvs(mission='K2', campaign=15, channel=24, cbv_indices='ALL')
    assert isinstance(cbvs, KeplerCotrendingBasisVectors)
    ax = cbvs.plot_cbvs('ALL')
    assert isinstance(ax, matplotlib.axes.Axes)

    cbvs = get_tess_cbvs(sector=10, camera=2, CCD=4, cbv_type = 'SingleScale', cbv_indices=np.arange(1,9))
    assert isinstance(cbvs, TessCotrendingBasisVectors)
    ax = cbvs.plot_cbvs([1,2,4,6,8])
    assert isinstance(ax, matplotlib.axes.Axes)

    cbvs = get_tess_cbvs(sector=10, camera=2, CCD=4, cbv_type = 'MultiScale', band=2, cbv_indices=np.arange(1,9))
    assert isinstance(cbvs, TessCotrendingBasisVectors)
    ax = cbvs.plot_cbvs('ALL')
    assert isinstance(ax, matplotlib.axes.Axes)


def test_compute_correlation():
    
    # Fully correlated matrix
    fluxMatrix = np.array([[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]])
    correlation_matrix = compute_correlation(fluxMatrix)
    assert np.all(correlation_matrix == 1.0)

    # Partially correlated
    fluxMatrix = np.array([[ 1.0, -1.0,  1.0, -1.0], 
                           [-1.0,  1.0,  1.0, -1.0], 
                           [ 1.0, -1.0,  1.0, -1.0], 
                           [-1.0,  1.0, -1.0,  1.0]])
    correlation_matrix = compute_correlation(fluxMatrix)
    correlation_truth = np.array([[ 1.0, -1.0,  0.5, -0.5], 
                           [-1.0,  1.0, -0.5,  0.5], 
                           [ 0.5, -0.5,  1.0, -1.0], 
                           [-0.5,  0.5, -1.0,  1.0]])
    assert_allclose(correlation_matrix, correlation_truth)

def test_CBVCorrector():

    # Create a CBVCorrector without reading CBVs from MAST
    sample_lc = TessLightCurve([1,2,3,4,5], [1,2,np.nan,4,5], [0.1, 0.1, 0.1, 0.1, 0.1],
            cadenceno=[1,2,3,4,5], flux_unit=u.Unit('electron / second'))

    cbvCorrector =  CBVCorrector(sample_lc, do_not_load_cbvs=True)
    # Check that Nan was removed
    assert len(cbvCorrector.lc.flux) == 4
    # Check that zero-centered median normalization occured
    assert_allclose(cbvCorrector._lc_median, 3.0)
    assert_allclose(np.mean(cbvCorrector.lc.flux), 0.0)

    dm = DesignMatrix(pd.DataFrame({'a':np.ones(4), 'b':[1,2,4,5]}))

    # Gaussian Prior fit
    lc = cbvCorrector.correct_gaussian_prior(cbv_type=None, cbv_indices=None,
        alpha=1e-9, ext_dm=dm)
    assert isinstance(lc, TessLightCurve)
    # Check that returned lc is in absolute flux units
    assert lc.flux_unit == u.Unit("electron / second")
    # The design matrix should have completely zeroed the flux around the median
    assert_allclose(lc.flux, cbvCorrector._lc_median)
    ax = cbvCorrector.diagnose()
    assert len(ax) == 2 and isinstance(ax[0], matplotlib.axes._subplots.Axes)

    # Now add a strong regularization term and under-fit the data
    lc = cbvCorrector.correct_gaussian_prior(cbv_type=None, cbv_indices=None,
        alpha=1e9, ext_dm=dm)
    # There should virtually no change in the flux
    assert_allclose(lc.flux, sample_lc.remove_nans().flux)

    # This should error because the dm has incorrect number of cadences
    dm_err = DesignMatrix(pd.DataFrame({'a':np.ones(5), 'b':[1,2,4,5,6]}))
    with pytest.raises(ValueError):
        lc = cbvCorrector.correct_gaussian_prior(cbv_type=None, cbv_indices=None,
            alpha=1e-2, ext_dm=dm_err)

    #***
    # ElasticNet fit
    lc = cbvCorrector.correct_ElasticNet(cbv_type=None, cbv_indices=None,
        alpha=1e-20, l1_ratio=0.5, ext_dm=dm)
    assert isinstance(lc, TessLightCurve) 
    assert lc.flux_unit == u.Unit("electron / second")
    # The design matrix should have completely zeroed the flux around the median
    assert_allclose(lc.flux, cbvCorrector._lc_median, rtol=1e-3)
    ax = cbvCorrector.diagnose()
    assert len(ax) == 2 and isinstance(ax[0], matplotlib.axes._subplots.Axes)
    # Now add a strong regularization term and under-fit the data
    lc = cbvCorrector.correct_ElasticNet(cbv_type=None, cbv_indices=None,
        alpha=1e9, l1_ratio=0.5, ext_dm=dm)
    # There should virtually no change in the flux
    assert_allclose(lc.flux, sample_lc.remove_nans().flux)


    #***
    # Correction optimizer
    # The optimizer cannot be run without downloading targest from MAST for use
    # within the under-fitting metric.
    # So let's just verify it fails as expected (not much else we can do)
    dm_err = DesignMatrix(pd.DataFrame({'a':np.ones(5), 'b':[1,2,4,5,6]}))
    with pytest.raises(ValueError):
        lc = cbvCorrector.correct_optimizer(cbv_type=None, cbv_indices=None, 
            alpha_bounds=[1e-4, 1e4], ext_dm=dm_err,
            target_over_score=0.5, target_under_score=0.8)

@pytest.mark.remote_data
def test_CBVCorrector_retrieval():
    """ Tests CBVCorrector by retrieving some sample Kepler/TESS light curves
    and correcting them
    """

    #***
    # A good TESS example of both over- and under-fitting
    # The "over-fitted" curve looks better to the eye, but eyes can be deceiving!
    lcf = search_lightcurve('TIC 357126143', mission='tess', sector=10).download()
    target = 'TIC 99180739'
    cbvCorrector =  CBVCorrector(lcf.SAP_FLUX)
    assert isinstance(cbvCorrector, CBVCorrector)

    cbv_type = ['SingleScale', 'Spike']
    cbv_indices = [np.arange(1,9), 'ALL']

    # Gaussian Prior correction
    lc = cbvCorrector.correct_gaussian_prior(cbv_type=cbv_type, cbv_indices=cbv_indices,
        alpha=1e-2)
    assert isinstance(lc, TessLightCurve) 
    # Check that returned lightcurve is in flux units
    assert lc.flux_unit == u.Unit("electron / second")
    ax = cbvCorrector.diagnose()
    assert len(ax) == 2 and isinstance(ax[0], matplotlib.axes._subplots.Axes)

    # ElasticNet corrections
    lc = cbvCorrector.correct_ElasticNet(cbv_type=cbv_type, cbv_indices=cbv_indices,
        alpha=1e1, l1_ratio=0.5)
    assert isinstance(lc, TessLightCurve) 
    assert lc.flux_unit == u.Unit("electron / second")
    ax = cbvCorrector.diagnose()
    assert len(ax) == 2 and isinstance(ax[0], matplotlib.axes._subplots.Axes)

    # Correction optimizer
    lc = cbvCorrector.correct_optimizer(cbv_type=cbv_type, cbv_indices=cbv_indices, 
        alpha_bounds=[1e-4, 1e4],
        target_over_score=0.5, target_under_score=0.8)
    assert isinstance(lc, TessLightCurve) 
    assert lc.flux_unit == u.Unit("electron / second")
    ax = cbvCorrector.diagnose()
    assert len(ax) == 2 and isinstance(ax[0], matplotlib.axes._subplots.Axes)

    # Goodness metric scan plot
    ax = cbvCorrector.goodness_metric_scan_plot(cbv_type=cbv_type, cbv_indices=cbv_indices)
    assert isinstance(ax, matplotlib.axes.Axes)

    # Try multi-scale basis vectors
    cbv_type = ['MultiScale.1', 'MultiScale.2', 'MultiScale.3']
    cbv_indices = ['ALL', 'ALL', 'ALL']
    lc = cbvCorrector.correct_gaussian_prior(cbv_type=cbv_type, cbv_indices=cbv_indices,
        alpha=1e-2)
    assert isinstance(lc, TessLightCurve) 
    
    #***
    # A Kepler and K2 example
    lcf = search_lightcurve('KIC 6508221', mission='kepler', quarter=5).download()
    cbvCorrector =  CBVCorrector(lcf.SAP_FLUX)
    lc = cbvCorrector.correct_gaussian_prior(alpha=1.0)
    assert isinstance(lc, KeplerLightCurve) 
    assert lc.flux_unit == u.Unit("electron / second")

    lcf = search_lightcurve('EPIC 247887989', mission='K2').download()
    cbvCorrector =  CBVCorrector(lcf.SAP_FLUX)
    lc = cbvCorrector.correct_gaussian_prior(alpha=1.0)
    assert isinstance(lc, KeplerLightCurve) 
    assert lc.flux_unit == u.Unit("electron / second")

    #***
    # Try some expected failures

    # cbv_type and cbv_indices not the same list lengths
    with pytest.raises(AssertionError):
        lc = cbvCorrector.correct_gaussian_prior(cbv_type=['SingleScale', 'Spike'], 
                cbv_indices=['all'], alpha=1e-2)

    # cbv_type is not a list
    with pytest.raises(AssertionError):
        lc = cbvCorrector.correct_gaussian_prior(cbv_type='SingleScale', 
                cbv_indices=['all'], alpha=1e-2)

    # cbv_indices is not a list
    with pytest.raises(AssertionError):
        lc = cbvCorrector.correct_gaussian_prior(cbv_type=['SingleScale'], 
                cbv_indices='all', alpha=1e-2)



