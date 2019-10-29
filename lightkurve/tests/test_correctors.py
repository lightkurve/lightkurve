from __future__ import division, print_function

import sys
import pytest

from numpy.testing import assert_almost_equal

from ..lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from ..lightcurvefile import KeplerLightCurveFile
from ..correctors import KeplerCBVCorrector, PLDCorrector
from ..search import search_targetpixelfile

from .test_lightcurve import TABBY_Q8

bad_optional_imports = False
try:
    import celerite
    import fbpca
except ImportError:
    bad_optional_imports = True

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
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_pld_corrector():
    # download tpf data for a target
    k2_target = 247887989
    k2_tpf = search_targetpixelfile(k2_target).download()
    # instantiate PLD corrector object
    pld = PLDCorrector(k2_tpf[:500])
    # produce a PLD-corrected light curve with a default aperture mask
    corrected_lc = pld.correct()
    # ensure the CDPP was reduced by the corrector
    pld_cdpp = corrected_lc.estimate_cdpp()
    raw_cdpp = k2_tpf.to_lightcurve().estimate_cdpp()
    assert(pld_cdpp < raw_cdpp)
    # make sure the returned object is the correct type (`KeplerLightCurve`)
    assert(isinstance(corrected_lc, KeplerLightCurve))
    # try detrending using a threshold mask
    corrected_lc = pld.correct(aperture_mask='threshold')
    # reduce using fewer principle components
    corrected_lc = pld.correct(n_pca_terms=20)
    # try PLD on a TESS observation
    from .. import TessTargetPixelFile
    from .test_targetpixelfile import TESS_SIM
    tess_tpf = TessTargetPixelFile(TESS_SIM)
    # instantiate PLD corrector object
    pld = PLDCorrector(tess_tpf[:500])
    # produce a PLD-corrected light curve with a pipeline aperture mask
    raw_lc = tess_tpf.to_lightcurve(aperture_mask='pipeline')
    corrected_lc = pld.correct(aperture_mask='pipeline', n_pca_terms=20,
                               use_gp=False)
    # the corrected light curve should have higher precision
    assert(corrected_lc.estimate_cdpp() < raw_lc.estimate_cdpp())
    # make sure the returned object is the correct type (`TessLightCurve`)
    assert(isinstance(corrected_lc, TessLightCurve))


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_to_corrector():
    """Does the tpf.pld() convenience method work?"""
    from .. import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    lc = tpf.to_corrector("pld").correct()
    assert len(lc.flux) == len(tpf.time)


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_pld_aperture_mask():
    """Test for #523: does PLDCorrector.correct() accept separate apertures for
    PLD pixels?"""
    from .. import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF
    tpf = KeplerTargetPixelFile(TABBY_TPF)
    # use only the pixels in the pipeline mask
    lc_pipeline = tpf.to_corrector("pld").correct(pld_aperture_mask='pipeline')
    # use all pixels in the tpf
    lc_all = tpf.to_corrector("pld").correct(pld_aperture_mask='all')
    # does this improve the correction?
    assert(lc_all.estimate_cdpp() < lc_pipeline.estimate_cdpp())
