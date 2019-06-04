import sys
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from astropy.utils.data import get_pkg_data_filename

from ..pymcpldcorrector import PyMCPLDCorrector as PLDCorrector

from ... import open
from ...search import search_targetpixelfile
from ...tests.test_targetpixelfile import filename_tpf_one_center, filename_tess

bad_optional_imports = False
try:
    import celerite
    import fbpca
except ImportError:
    bad_optional_imports = True

@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_first_order_matrix():
    """Test the basic construction of the 1st order regressor matrix."""
    # Open a 3x3 TPF which has flux=1 in the center pixel and 0 elsewhere
    tpf = open(filename_tpf_one_center)
    # Try with all pixels in the mask
    corr = PLDCorrector(tpf, pld_aperture_mask="all")
    matrix = corr.create_first_order_matrix()
    assert matrix.shape == (len(tpf.time), 9)
    assert np.sum(matrix) == len(tpf.time)
    # Only include central pixel
    corr = PLDCorrector(tpf, pld_aperture_mask=(tpf.flux[0] > 0))
    matrix = corr.create_first_order_matrix()
    assert matrix.shape == (len(tpf.time), 1)
    assert np.sum(matrix) == len(tpf.time)


@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_design_matrix():
    """Test the basic construction of the PLD design matrix."""
    # Open a 3x3 TPF which has flux=1 in the center pixel and 0 elsewhere
    tpf = open(filename_tpf_one_center)
    n_pixels = tpf.flux[0].size
    corr = PLDCorrector(tpf, pld_aperture_mask="all")
    # Does the design matrix shape increase as expected for higher-order PLD?
    for pld_order in [1, 2, 3]:
        for n_pca_terms in [1, 3]:
            matrix = corr.create_design_matrix(pld_order=pld_order, n_pca_terms=n_pca_terms)
            assert matrix.shape == (len(tpf.time), n_pixels + n_pca_terms*(pld_order - 1))


@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_pymc_model():
    tpf = open(filename_tess)
    corr = PLDCorrector(tpf, pld_aperture_mask="all")
    model = corr.create_pymc_model(design_matrix=corr.create_design_matrix(pld_order=1))
    sol = corr.optimize(model)


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
