import pytest
import celerite
import numpy as np

from ... import search_targetpixelfile
from .. import PLDCorrector

bad_optional_imports = False
try:
    import celerite
    import fbpca
except ImportError:
    bad_optional_imports = True

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
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
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_custom_gp_params():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    # can we make GP objects with custom kernels?
    clc = pld.correct(kernel="sho")
    # can we pass in our own celerite GP kernel?
    log_omega0=np.log(2*np.pi / 30)
    log_S0=np.log(10000)
    log_Q=np.log(10)
    kernel = celerite.terms.SHOTerm(log_omega0=log_omega0, log_S0=log_S0, log_Q=log_Q,
                                    bounds={'log_S0': (-2 + log_S0, 2 + log_S0),
                                            'log_Q': (0.2, 7),
                                            'log_w0': (np.log(2*np.pi/150), np.log(2*np.pi/0.1))})
    clc = pld.correct(kernel=kernel)

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_correct():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    # does the correct function run smoothly?
    clc = pld.correct()
    # does it improve the cdpp?
    assert clc.estimate_cdpp() < tpf.to_lightcurve().estimate_cdpp()

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="PLD requires celerite and fbpca")
def test_diagnose():
    tpf = search_targetpixelfile('k2-199')[0].download()
    pld = PLDCorrector(tpf)
    clc = pld.correct()
    # does the diagnose function run smoothly?
    pld.diagnose()
