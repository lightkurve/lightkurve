import pytest
import celerite

from ... import search_targetpixelfile
from .. import PLDCorrector # change this lol

@pytest.mark.remote_data
def test_create_design_matrix():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    # can we make a first order matrix?
    fo_matrix = pld.create_design_matrix(pld_order=1)
    # should have shape (n_cadences, n_pca_terms)
    assert(fo_matrix.shape == (len(tpf.flux), 10))
    # second order?
    so_matrix = pld.create_design_matrix(pld_order=2, n_pca_terms=8)
    assert so_matrix.shape == (len(tpf.flux), 16)

@pytest.mark.remote_data
def test_gp_model():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    # can we create a celerite GP object for our light curve?
    gp = pld.create_gp_model()
    assert isinstance(gp, celerite.GP)
    # can we optimize our GP for a given design matrix?
    matrix = pld.create_design_matrix(pld_order=1)
    soln = pld.optimize(matrix)

@pytest.mark.remote_data
def test_correct():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    # does the correct function run smoothly?
    clc = pld.correct()
    # does it improve the cdpp?
    assert clc.estimate_cdpp() < tpf.to_lightcurve().estimate_cdpp()

@pytest.mark.remote_data
def test_diagnose():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    clc = pld.correct()
    # does the diagnose function run smoothly?
    pld.diagnose()
