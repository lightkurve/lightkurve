from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal

from ..lightcurve import LightCurve
from ..convenience import cdpp


def test_cdpp():
    """Tests the cdpp() convenience function which wraps `LightCurve.cdpp()`"""
    flux = np.random.normal(loc=1, scale=100e-6, size=10000)
    lc = LightCurve(np.arange(10000), flux)
    assert_almost_equal(cdpp(flux), lc.cdpp())
