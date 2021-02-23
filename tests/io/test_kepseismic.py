import pytest

from astropy.io import fits
import numpy as np

from lightkurve.io.kepseismic import read_kepseismic_lightcurve
from lightkurve.io.detect import detect_filetype

@pytest.mark.remote_data
def test_detect_kepseismic():
    """Can we detect the correct format for KEPSEISMIC files?"""
    url = "https://archive.stsci.edu/hlsps/kepseismic/001200000/92147/20d-filter/hlsp_kepseismic_kepler_phot_kplr001292147-20d_kepler_v1_cor-filt-inp.fits"
    f = fits.open(url)

    assert detect_filetype(f) == "KEPSEISMIC"


@pytest.mark.remote_data
def test_read_kepseismic():
    """Can we read KEPSEISMIC files?"""
    url = "https://archive.stsci.edu/hlsps/kepseismic/001200000/92147/20d-filter/hlsp_kepseismic_kepler_phot_kplr001292147-20d_kepler_v1_cor-filt-inp.fits"
    with fits.open(url, mode="readonly") as hdulist:
        fluxes = hdulist[1].data["FLUX"]

    lc = read_kepseismic_lightcurve(url)

    flux_lc = lc.flux.value

    # print(flux_lc, fluxes)
    assert np.sum(fluxes) == np.sum(flux_lc)
