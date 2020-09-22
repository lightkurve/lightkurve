""" cbvcorrector.py module unit tests
"""

import pytest
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_allclose, assert_raises)

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
from astropy.table import Table
from astropy.time import Time

from ... import TessLightCurve, KeplerLightCurve
from ... import search_lightcurve
from ... import LightkurveWarning
from ..designmatrix import DesignMatrix
from ..cbvcorrector import download_kepler_cbvs, download_tess_cbvs, \
    CotrendingBasisVectors, KeplerCotrendingBasisVectors, \
    TessCotrendingBasisVectors


def test_CotrendingBasisVectors_nonretrieval():
    """ Tests CotrendingBasisVectors class without requiring remote data
    """

    #***
    # Constructor
    # Create some generic CotrendingBasisVectors objects

    # Generic CotrendingBasisVectors object
    dataTbl = Table([[1, 2, 3], [False, True, False], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], 
        names=('CADENCENO', 'GAP', 'VECTOR_1', 'VECTOR_3'))
    cbvTime = Time([443.51090033, 443.53133457, 443.55176891], format='bkjd')
    cbvs = CotrendingBasisVectors(data=dataTbl, time=cbvTime)
    assert isinstance(cbvs, CotrendingBasisVectors)
    assert cbvs.cbv_indices == [1, 3]
    assert np.all(cbvs.time.value == [443.51090033, 443.53133457, 443.55176891])
    
    # Auto-initiate 'GAP' and 'CADENCENO'
    dataTbl = Table([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], 
            names=('VECTOR_3', 'VECTOR_12'))
    cbvTime = Time([443.51090033, 443.53133457, 443.55176891], format='bkjd')
    cbvs = CotrendingBasisVectors(data=dataTbl, time=cbvTime)
    assert isinstance(cbvs, CotrendingBasisVectors)
    assert cbvs.cbv_indices == [3, 12]
    assert np.all(cbvs.gap_indicators == [False, False, False])
    assert np.all(cbvs.cadenceno == [0, 1, 2])

    
    #***
    # _to_designmatrix
    # Make sure CBVs are the columns in the returned 2-dim array
    dataTbl = Table([[1, 2, 3], [False, True, False], 
            [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 
            names=('CADENCENO', 'GAP', 'VECTOR_1', 'VECTOR_2', 'VECTOR_3'))
    cbvTime = Time([1569.44053967, 1569.44192856, 1569.44331746], format='btjd')
    cbvs = CotrendingBasisVectors(dataTbl, cbvTime)
    cbv_dm_name = 'test cbv set'
    # CBV index 5 does not exists and should be ingored
    cbv_designmatrix = cbvs.to_designmatrix(cbv_indices=[1,3,5], name=cbv_dm_name)
    assert cbv_designmatrix.shape == (3,2)
    assert np.all(cbv_designmatrix['VECTOR_1'] == np.array([1.0, 2.0, 3.0]))
    assert np.all(cbv_designmatrix['VECTOR_3'] == np.array([7.0, 8.0, 9.0]))
    assert cbv_designmatrix.name == cbv_dm_name
    # CBV #2 was not requested, so make sure it is not present
    with pytest.raises(KeyError):
        cbv_designmatrix['VECTOR_2']
    
    #***
    # plot
    ax = cbvs.plot(cbv_indices=[1,2], ax=None)
    assert isinstance(ax, matplotlib.axes.Axes)
    
    # There is no CBV # 5 so the third cbv_indices entry will be ignored
    ax = cbvs.plot(cbv_indices=[1,2,5], ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
    
    # CBVs use 1-based indexing. Throw error if requesting CBV index 0
    with pytest.raises(ValueError):
        ax = cbvs.plot(cbv_indices=[0,1,2], ax=ax)

    # Only 'all' or specific CBV indices can be requested
    with pytest.raises(ValueError):
        ax = cbvs.plot('Doh!')

    
    #***
    # align
    # Set up some cadenceno such that both CBV is trimmed and NaNs inserted
    sample_lc = TessLightCurve(time=[1,2,3,4,6,7], flux=[1,2,3,4,6,7],
            flux_err=[0.1,0.1, 0.1, 0.1, 0.1, 0.1], cadenceno=[1,2,3,4,6,7])
    dataTbl = Table([[1, 2, 3, 5, 6], [False, True, False, False, False], 
            [1.0, 2.0, 3.0, 5.0, 6.0 ]], 
        names=('CADENCENO', 'GAP', 'VECTOR_1'))
    cbvTime = Time([1569.43915078, 1569.44053967, 1569.44192856,
           1569.44470635, 1569.44609524], format='btjd')
    cbvs = CotrendingBasisVectors(dataTbl, cbvTime)
    cbvs = cbvs.align(sample_lc)
    assert np.all(sample_lc.cadenceno == cbvs.cadenceno)
    assert len(cbvs.cadenceno) == 6
    assert len(sample_lc.flux) == 6
    assert np.all(cbvs.gap_indicators.value[[1,3,5]])
    # Ignore the warning in to_designmatric due to a low rank matrix
    with warnings.catch_warnings():
        # Instantiating light curves with NaN times will yield a warning
        warnings.simplefilter("ignore", LightkurveWarning)
        cbv_designmatrix = cbvs.to_designmatrix(cbv_indices=[1])
    assert np.all(cbv_designmatrix['VECTOR_1'][[0,1,2,4]] == [1.0, 2.0, 3.0, 6.0])
    assert np.all(np.isnan(cbv_designmatrix['VECTOR_1'][[3,5]]))

    #***
    # interpolate
    nLcCadences = 20
    xLc = np.linspace(0.0, 2*np.pi, num=nLcCadences)
    sample_lc = TessLightCurve(time=xLc, flux=np.sin(xLc), flux_err=np.full(nLcCadences, 0.1),
            cadenceno=np.arange(nLcCadences))
    nCbvCadences = 10
    xCbv = np.linspace(0.0, 2*np.pi, num=nCbvCadences)
    dataTbl = Table([np.arange(nCbvCadences), np.full(nCbvCadences, False), 
            np.cos(xCbv), np.sin(xCbv+np.pi*0.125)], 
            names=('CADENCENO', 'GAP', 'VECTOR_1', 'VECTOR_2'))
    cbvTime = Time(xCbv, format='btjd')
    cbvs = CotrendingBasisVectors(dataTbl, cbvTime)
    cbv_interpolated = cbvs.interpolate(sample_lc, extrapolate=False)
    assert np.all(cbv_interpolated.time.value == sample_lc.time.value)
    # Extrapolation test
    xCbv = np.linspace(0.0, 1.5*np.pi, num=nCbvCadences)
    dataTbl = Table([np.arange(nCbvCadences), np.full(nCbvCadences, False), 
            np.cos(xCbv), np.sin(xCbv+np.pi*0.125)], 
            names=('CADENCENO', 'GAP', 'VECTOR_1', 'VECTOR_2'))
    cbvTime = Time(xCbv, format='btjd')
    cbvs = CotrendingBasisVectors(dataTbl, cbvTime)
    cbv_interpolated = cbvs.interpolate(sample_lc, extrapolate=False)
    assert np.all(np.isnan(
        cbv_interpolated['VECTOR_1'].value[
            np.nonzero(cbv_interpolated.time.value > 1.5*np.pi)[0]]))
    cbv_interpolated = cbvs.interpolate(sample_lc, extrapolate=True)
    assert np.all(np.logical_not(np.isnan(
        cbv_interpolated['VECTOR_1'].value[
            np.nonzero(cbv_interpolated.time.value > 1.5*np.pi)[0]])))
    
@pytest.mark.remote_data
def test_cbv_retrieval():
    """ Tests reading in some CBVs from MAST

        This indirectly tests the classes KeplerCotrendingBasisVectors and
        TessCotrendingBasisVectors

    """

    cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, cbv_type = 'SingleScale')
    assert isinstance(cbvs, TessCotrendingBasisVectors)
    ax = cbvs.plot([1,2,4,6,8])
    assert isinstance(ax, matplotlib.axes.Axes)
    assert cbvs.mission == 'TESS'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.band is None
    assert cbvs.sector == 10
    assert cbvs.camera == 2
    assert cbvs.ccd == 4
    
    cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, cbv_type = 'MultiScale', band=2)
    assert isinstance(cbvs, TessCotrendingBasisVectors)
    ax = cbvs.plot('all')
    assert isinstance(ax, matplotlib.axes.Axes)
    assert cbvs.band == 2
    
    cbvs = download_tess_cbvs(sector=8, camera=3, ccd=1, cbv_type = 'Spike')
    assert isinstance(cbvs, TessCotrendingBasisVectors)
    ax = cbvs.plot('all')
    assert isinstance(ax, matplotlib.axes.Axes)
    
    # No band specified for MultiScale, this should error
    with pytest.raises(AssertionError):
        cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, cbv_type = 'MultiScale')
    # Band specified for SingleScale, this should also error
    with pytest.raises(AssertionError):
        cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, cbv_type = 'SingleScale', band=2)
    # Improper CBV type request
    with pytest.raises(Exception):
        cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, cbv_type = 'SuperSingleScale')

    cbvs = download_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4)
    assert isinstance(cbvs, KeplerCotrendingBasisVectors)
    ax = cbvs.plot('all')
    assert isinstance(ax, matplotlib.axes.Axes)
    assert cbvs.mission == 'Kepler'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.quarter == 8
    assert cbvs.campaign is None
    assert cbvs.module == 16
    assert cbvs.output == 4

    cbvs = download_kepler_cbvs(mission='K2', campaign=15, channel=24)
    assert isinstance(cbvs, KeplerCotrendingBasisVectors)
    ax = cbvs.plot('all')
    assert isinstance(ax, matplotlib.axes.Axes)
    assert cbvs.mission == 'K2'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.quarter is None
    assert cbvs.campaign == 15
    assert cbvs.module == 8
    assert cbvs.output == 4
