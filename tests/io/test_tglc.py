import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.detect import detect_filetype
from lightkurve.io.tglc import read_tglc_lightcurve


@pytest.mark.remote_data
def test_tglc():
    """Can we read in TGLC light curves?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tglc/s0001/cam4-ccd2/0046/2474/2688/9442/hlsp_tglc_tess_ffi_gaiaid-4624742688944261376-s0001-cam4-ccd2_tess_v1_llc.fits"
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a TGLC file?
        assert detect_filetype(hdulist) == "TGLC"
        # Are the correct fluxes read in?
        lc = read_tglc_lightcurve(url, quality_bitmask=0)
        assert lc.meta["AUTHOR"] == "TGLC"
        assert lc.meta["FLUX_ORIGIN"] == "cal_psf_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["cal_psf_flux"])
        assert np.issubdtype(lc["cadenceno"].dtype, np.integer)


@pytest.mark.remote_data
def test_search_tglc():
    """Can we search and download a TGLC light curve?"""
    # Try an early campaign
    search = search_lightcurve("TIC 140898436", author="TGLC", sector=1, mission="TESS")
    assert len(search) == 1
    assert search.table["author"][0] == "TGLC"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.targetid == 140898436
    assert lc.sector == 1
    assert lc.camera == 4
    assert lc.ccd == 2
