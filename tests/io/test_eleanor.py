import pytest

from astropy.io import fits
from astropy import units as u
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.eleanor import read_eleanor_lightcurve
from lightkurve.io.detect import detect_filetype


@pytest.mark.remote_data
def test_read_eleanor():
    """Can we read in ELEANOR / GSFC-ELEANOR-LITE light curves?"""
    # pi Men in sector 1
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/gsfc-eleanor-lite/s0001/0000/0002/6113/6679/hlsp_gsfc-eleanor-lite_tess_ffi_s0001-0000000261136679_tess_v1.0_lc.fits"
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a QLP file?
        assert detect_filetype(hdulist) == "ELEANOR"
        # Are the correct fluxes read in?
        lc = read_eleanor_lightcurve(url)
        assert lc.meta["FLUX_ORIGIN"] == "corr_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["CORR_FLUX"])
        assert lc.flux.unit == u.electron / u.second
        assert np.isnan(lc.flux_err.value).all()

        # Are the correct fluxes read in? case raw flux
        lc = read_eleanor_lightcurve(url, flux_column="raw_flux")
        assert lc.meta["FLUX_ORIGIN"] == "raw_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["RAW_FLUX"])
        assert lc.flux.unit == u.electron / u.second
        assert_array_equal(lc.flux_err.value, hdulist[1].data["FLUX_ERR"])
        assert lc.flux_err.unit == u.electron / u.second

        # assert misc metadata compatibility fixes
        assert lc.meta["AUTHOR"] == "ELEANOR"
        assert lc.meta["TARGETID"] == 261136679
        assert lc.meta["TICID"] == 261136679
        assert lc.meta["OBJECT"] == "TIC 261136679"
        assert lc.meta["LABEL"] == "TIC 261136679"


@pytest.mark.remote_data
def test_search_gsfc_eleanor_lite():
    """Can we search and download GSFC-ELEANOR-LITE light curves from MAST?"""
    search = search_lightcurve("TIC 261136679", author="GSFC-ELEANOR-LITE")
    assert len(search) > 0
    assert search.table["author"][0] == "GSFC-ELEANOR-LITE"
    lc = search[0].download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == search[0].table["sequence_number"]
    assert lc.author == "ELEANOR"
