import numpy as np

from ... import open
from ..pymcpldcorrector import PyMCPLDCorrector as PLDCorrector

from ...tests.test_targetpixelfile import filename_tpf_one_center


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
