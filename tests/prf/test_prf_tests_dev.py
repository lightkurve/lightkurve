import lightkurve as lk
from astropy.utils.data import get_pkg_data_filename
import numpy as np

filename_tess = get_pkg_data_filename("../data/tess25155310-s01-first-cadences.fits.gz")
# a local version of TABBY_TPF with ~ 2 days of data; should be sufficient for most tests
filename_tpf_tabby_lite = get_pkg_data_filename(
    "../data/test-tpf-kplr-tabby-100-cadences.fits"
)


# Check if a Kepler TPF has a prf attribute
# Check if tpf.prf returns an object of class PRF (specifically KeplerPRF)
# Check that the shape of the PRF object matches the TPF
# Check that the sum of the vanilla PRF values is very close to 1 
def test_kepler():
    tpf = lk.KeplerTargetPixelFile(filename_tpf_tabby_lite)

    # Check that the TargetPixelFile class has the appropriate prf functions
    assert hasattr(tpf, "prf")
    assert isinstance(tpf.prf, lk.prf.prfmodel.PRF)
    assert isinstance(tpf.prf, lk.prf.prfmodel.KeplerPRF)
    assert tpf.prf.shape == tpf.shape[1:3]
    assert np.isclose(np.sum(tpf.prf.prf_model()), 1.0, rtol=0.01)
    # Scaling will cause some of the flux to extend out of the TPF pixels
    assert np.sum(tpf.prf.prf_model()) > np.sum(tpf.prf.prf_model(scale=3))
    
    # Check that prf.from_tpf() works
    prf_from_tpf = lk.prf.from_tpf(tpf)

    assert isinstance(prf_from_tpf, lk.prf.prfmodel.PRF)
    assert np.isclose(np.sum(prf_from_tpf.prf_model()), 1.0, rtol=0.01)

    ################# APERTURE CHECKS ##################
    # Check that tpf.simple_aperture returns an aperture of the same shape
    # Check the aperture contains at least one pixel
    # Check that there are more pixels in apertures containing more flux
    aperture = tpf.get_simple_aperture(min_completeness=0.9)
    assert aperture.shape == tpf.shape[1:3]
    assert np.sum(aperture) >= 1
    assert np.sum(aperture) > np.sum(tpf.get_simple_aperture(min_completeness=0.1))

    # Running this with only 1 target star should be default always show no contamination
    # assert tpf.estimate_contamination(aperture) == 1.

    # Create a Kepler PRF directly (ie not tied to a TFP)
    
    kep_prf = lk.prf.KeplerPRF(column=100, row=200, channel=6, shape=(15,15))
    kep_model = kep_prf.prf_model(center_col=[107,106], center_row=[207,204])
    assert kep_model.shape == (2,15,15)
    #assert np.isclose(np.sum(kep_model[0,:,:]), 1.0, rtol=0.01)
    
    # Create a Kepler model near the edge of the CCD, which should zero out the pixels
    kep_prf = lk.prf.KeplerPRF(column=0, row=0, channel=6, shape=(10,10))
    kep_model = kep_prf.prf_model(center_col=5, center_row=5)
    assert kep_model.shape == (1,10,10)
    assert np.sum(kep_model== 0)


    # Check if a TESS TPF has a prf attribute and that it returns an object of class PRF
def test_tess():
    tpf = lk.TessTargetPixelFile(filename_tess)
    # Check that the TargetPixelFile class has the appropriate prf functions
    assert hasattr(tpf, "prf")
    assert isinstance(tpf.prf, lk.prf.prfmodel.PRF)
    assert isinstance(tpf.prf, lk.prf.prfmodel.TessPRF)
    assert tpf.prf.shape == tpf.shape[1:3]
    assert np.isclose(np.sum(tpf.prf.prf_model()), 1.0, rtol=0.01)

    ################# APERTURE CHECKS ##################
    # Check that tpf.simple_aperture returns an aperture of the same shape
    # Check the aperture contains at least one pixel
    # Check that there are more pixels in apertures containing more flux
    aperture = tpf.get_simple_aperture(min_completeness=0.9)
    assert aperture.shape == tpf.shape[1:3]
    assert np.sum(aperture) >= 1
    assert np.sum(aperture) > np.sum(tpf.get_simple_aperture(min_completeness=0.1))
    
    test_prf = lk.prf.TessPRF(column=100, row=200, camera=4, ccd=2, shape=(15,15))
    assert test_prf.prf_model(center_col=np.arange(100,115, 2), center_row=np.arange(100,115, 2)).shape == (len(np.arange(100,115, 2)), 15,15)

