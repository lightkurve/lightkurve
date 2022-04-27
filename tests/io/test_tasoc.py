import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.tasoc import read_tasoc_lightcurve
from lightkurve.io.detect import detect_filetype


# The URL needs to be updated upon a new TASOC data release.
TEST_TIC_ID = 150441810
TEST_FIT_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tasoc/s0001/c1800/0000/0001/5044/1810/hlsp_tasoc_tess_ffi_tic00150441810-s0001-cam4-ccd4-c1800_tess_v05_ens-lc.fits"


@pytest.mark.remote_data
def test_detect_tasoc():
    """Can we detect the correct format for TASOC files?"""
    url = TEST_FIT_URL
    f = fits.open(url)

    assert detect_filetype(f) == "TASOC"


@pytest.mark.remote_data
def test_read_tasoc():
    """Can we read TASOC files?"""
    url = TEST_FIT_URL
    with fits.open(url, mode="readonly") as hdulist:
        fluxes = hdulist[1].data["FLUX_RAW"]

    lc = read_tasoc_lightcurve(url, flux_column="FLUX_RAW")

    assert lc.meta["FLUX_ORIGIN"] == "flux_raw"
    assert_array_equal(fluxes, lc.flux.value)


@pytest.mark.remote_data
def test_search_tasoc():
    """Can we search and download a TASOC light curve?"""
    search = search_lightcurve(f"TIC {TEST_TIC_ID}", author="TASOC")
    assert len(search) >= 1
    assert search.table["author"][0] == "TASOC"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
