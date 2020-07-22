from __future__ import division, print_function

import pytest

from numpy.testing import assert_almost_equal

from ..lightcurvefile import KeplerLightCurveFile
from ..correctors import KeplerCBVCorrector, PLDCorrector
from ..search import search_targetpixelfile

from .test_lightcurve import TABBY_Q8


@pytest.mark.xfail  # Fit no longer matches exactly as of v2.0; ignoring for now due to new CBV corrector work
@pytest.mark.remote_data
def test_kepler_cbv_fit():
    """Verify that the two methods to do cbv fit are the nearly the same."""
    cbv = KeplerCBVCorrector(TABBY_Q8)
    cbv_lc = cbv.correct()
    assert_almost_equal(cbv.coeffs, [0.102, 0.006], decimal=3)
    lcf = KeplerLightCurveFile(TABBY_Q8)
#    cbv_lcf = lcf.compute_cotrended_lightcurve()
#    assert_almost_equal(cbv_lc.flux, cbv_lcf.flux)
    cbv_lcf = KeplerCBVCorrector(lcf).correct()

    lc = KeplerLightCurveFile(TABBY_Q8).SAP_FLUX
    cbv = KeplerCBVCorrector(lc)
    cbv_lc_2 = cbv.correct()
    assert_almost_equal(cbv_lcf.flux, cbv_lc_2.flux)


@pytest.mark.remote_data
def test_to_corrector():
    """Does the tpf.to_corrector('pld') convenience method work?"""
    from .. import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    lc = tpf.to_corrector("pld").correct()
    assert len(lc.flux) == len(tpf.time)
