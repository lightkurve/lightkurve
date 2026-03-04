import pytest
import numpy as np

TABBY_FAST = (
    "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS"
    "/product/tess2021204101404-s0041-0000000185336364-0212-a_fast-lc.fits"
)

TABBY_REG = (
    "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:TESS"
    "/product/tess2021204101404-s0041-0000000185336364-0212-s_lc.fits"
)


@pytest.mark.remote_data
def test_to_corrector():
    """Does the tpf.to_corrector('pld') convenience method work?"""
    from lightkurve import KeplerTargetPixelFile
    from .test_targetpixelfile import TABBY_TPF

    tpf = KeplerTargetPixelFile(TABBY_TPF)
    lc = tpf.to_corrector("pld").correct()
    assert len(lc.flux) == len(tpf.time)

@pytest.mark.remote_data
def test_CBV_cadence():
    """Separate CBVs are obtained for 2-min and 20-s data products.
    Check that the relevant CBVs are found."""
    from lightkurve import TessLightCurveFile
    from lightkurve.correctors import CBVCorrector
    lc_fast = TessLightCurveFile(TABBY_FAST)
    fast_cbvs = CBVCorrector(lc_fast)
    # assert the CBVs have a cadence of 20-s for the fast data product
    assert np.isclose(np.median((fast_cbvs.cbvs[0]['time'][1:] - fast_cbvs.cbvs[0]['time'][:-1]).to('second').value), 20, .01)

    # assert the CBVs have a cadence of 2-min for the regular lightcurve product
    lc = TessLightCurveFile(TABBY_REG)
    cbvs = CBVCorrector(lc)
    assert np.isclose(np.median((cbvs.cbvs[0]['time'][1:] - cbvs.cbvs[0]['time'][:-1]).to('second').value), 120, .01)