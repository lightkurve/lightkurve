from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal

from ..lightcurve import LightCurve
from ..convenience import estimate_cdpp


def test_cdpp():
    """Tests the estimate_cdpp() convenience function which wraps
    `LightCurve.estimate_cdpp()`"""
    flux = np.random.normal(loc=1, scale=100e-6, size=10000)
    lc = LightCurve(time=np.arange(10000), flux=flux)
    assert_almost_equal(estimate_cdpp(flux), lc.estimate_cdpp())
