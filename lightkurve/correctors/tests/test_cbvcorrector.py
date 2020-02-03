""" cbvcorrector.py module unit tests
"""

import pytest
#from numpy.testing import (assert_almost_equal, assert_array_equal,
#                           assert_allclose, assert_raises)

import numpy as np
import matplotlib.pyplot as plt

from .. import DEFAULT_NUMBER_CBVS
from .. import get_cbvs, CotrendingBasisVectors

def test_cbv_nonretrieval():
    """ Tests that do not require remote_data
    """

    # Create some simple CotrendingBasisVectors objects
    desired_indices = [1,2,4,6,8]
    cbvs = CotrendingBasisVectors('Kepler', cbv_type='SingleScale',
            cbv_indices=desired_indices)
    assert cbvs.mission == 'Kepler'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.cbv_indices == desired_indices
    assert cbvs.band is None

    desired_indices = [1,2,4,6,8]
    cbvs = CotrendingBasisVectors('Kepler', cbv_type='SingleScale',
            cbv_indices='ALL')
    assert cbvs.mission == 'Kepler'
    assert cbvs.cbv_type == 'SingleScale'
    assert cbvs.cbv_indices == np.arange(1,DEFAULT_NUMBER_CBVS+1)
    assert cbvs.band is None

    # These should fail
    with pytest.raises(ValueError):
        CotrendingBasisVectors('SuperKepler', cbv_type='SingleScale',
            cbv_indices=desired_indices)
    with pytest.raises(ValueError):
        CotrendingBasisVectors('Kepler', cbv_type='SuperSingleScale',
            cbv_indices=desired_indices)


#@pytest.mark.remote_data
def test_cbv_retrieval():
    """
    """
    cbvs = get_cbvs(mission='Kepler', quarter=8, module=16, output=4, cbv_indices=np.arange(1,7))
    cbvs.plot_cbvs('ALL')
    plt.show()

    cbvs = get_cbvs(mission='K2', campaign=15, channel=24, cbv_indices='ALL')
    cbvs.plot_cbvs('ALL')
    plt.show()

    cbvs = get_cbvs(mission='TESS', sector=10, camera=2, CCD=4, cbv_type = 'SingleScale', cbv_indices=np.arange(1,9))
    cbvs.plot_cbvs([1,2,4,6,8])
    plt.show()

    cbvs = get_cbvs(mission='TESS', sector=10, camera=2, CCD=4, cbv_type = 'MultiScale', band=2, cbv_indices=np.arange(1,9))
    cbvs.plot_cbvs('ALL')
    plt.show()


