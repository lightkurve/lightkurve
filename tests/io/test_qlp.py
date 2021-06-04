import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.qlp import read_qlp_lightcurve
from lightkurve.io.detect import detect_filetype


@pytest.mark.remote_data
def test_qlp():
    """Can we read in QLP light curves?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/qlp/s0011/0000/0002/7755/4109/hlsp_qlp_tess_ffi_s0011-0000000277554109_tess_v01_llc.fits"    
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a QLP file?
        assert detect_filetype(hdulist) == "QLP"
        # Are the correct fluxes read in?
        lc = read_qlp_lightcurve(url, quality_bitmask=0)
        assert lc.meta["FLUX_ORIGIN"] == "sap_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["SAP_FLUX"])


@pytest.mark.remote_data
def test_search_qlp():
    """Can we search and download QLP light curves from MAST?"""
    search = search_lightcurve("TIC 277554109", author="QLP", sector=11)
    assert len(search) == 1
    assert search.table["author"][0] == "QLP"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == 11
    assert lc.author == "QLP"
