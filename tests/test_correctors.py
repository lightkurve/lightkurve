from __future__ import division, print_function

import pytest

from numpy.testing import assert_almost_equal

from ..lightcurvefile import KeplerLightCurveFile
from ..search import search_targetpixelfile

from .test_lightcurve import TABBY_Q8

@pytest.mark.remote_data
def test_to_corrector():
    """Does the tpf.to_corrector('pld') convenience method work?"""
    from .. import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    lc = tpf.to_corrector("pld").correct()
    assert len(lc.flux) == len(tpf.time)
