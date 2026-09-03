import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.detect import detect_filetype
from lightkurve.io.tars import read_tars_lightcurve


@pytest.mark.remote_data
def test_tars():
    """Can we read in TARS light curves?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tars/sec_55_cam_1_ccd_3/hlsp_tars_tess_ffi_s0055-0000000051791910_tess_v01_lc.fits"
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a TARS file?
        assert detect_filetype(hdulist) == "TARS"
        # Are the correct fluxes read in?
        lc = read_tars_lightcurve(url, quality_bitmask=0)
        assert lc.meta["AUTHOR"] == "TARS"
        assert lc.meta["FLUX_ORIGIN"] == "flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["FLUX"])
        # TARS files carry no quality flags; a zero-filled column is assigned
        assert_array_equal(lc["quality"], np.zeros(len(lc), dtype=np.int32))


@pytest.mark.remote_data
def test_search_tars():
    """Can we search and download a TARS light curve?"""
    search = search_lightcurve("TIC 51791910", author="TARS", sector=55, mission="TESS")
    assert len(search) == 1
    assert search.table["author"][0] == "TARS"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.targetid == 51791910
    assert lc.sector == 55
    assert lc.camera == 1
    assert lc.ccd == 3
