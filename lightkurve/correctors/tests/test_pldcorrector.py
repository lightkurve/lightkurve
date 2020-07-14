import pytest

from astropy.utils.data import get_pkg_data_filename
import numpy as np

from ... import search_targetpixelfile, KeplerLightCurve, TessLightCurve
from .. import PLDCorrector, KeplerPLDCorrector, TessPLDCorrector

@pytest.mark.remote_data
def test_kepler_pld_corrector():
    tpf = search_targetpixelfile('K2-199')[0].download()
    pld = PLDCorrector(tpf)
    # Is the correct filetype returned?
    clc = pld.correct()
    assert(isinstance(clc, KeplerLightCurve))
    # Does the diagnostic plot run?
    pld.diagnose()
    # Does sparse correction work?
    pld.correct(sparse=True)
    # Did the correction with default values help?
    raw_lc = tpf.to_lightcurve(aperture_mask='threshold')
    assert(clc.estimate_cdpp() < raw_lc.estimate_cdpp())

@pytest.mark.remote_data
def test_tess_pld_corrector():
    tpf = search_targetpixelfile('TOI 700')[0].download()
    pld = PLDCorrector(tpf)
    # Is the correct filetype returned?
    clc = pld.correct()
    assert(isinstance(clc, TessLightCurve))
    # Does the diagnostic plot run?
    pld.diagnose()
    pld.correct(sparse=True)
    # Did the correction with default values help?
    raw_lc = tpf.to_lightcurve(aperture_mask='threshold')
    assert(clc.estimate_cdpp() < raw_lc.estimate_cdpp())

@pytest.mark.remote_data
def test_pld_aperture_mask():
    """Test for #523: does PLDCorrector.correct() accept separate apertures for
    PLD pixels?"""
    tpf = search_targetpixelfile('K2-199')[0].download()
    # use only the pixels in the pipeline mask
    lc_pipeline = tpf.to_corrector("pld").correct(pld_aperture_mask='pipeline')
    # use all pixels in the tpf
    lc_all = tpf.to_corrector("pld").correct(pld_aperture_mask='all')
    # does this improve the correction?
    assert(lc_all.estimate_cdpp() < lc_pipeline.estimate_cdpp())
