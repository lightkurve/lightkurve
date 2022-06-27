import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.eleanorlite import read_eleanorlite_lightcurve
from lightkurve.io.detect import detect_filetype

@pytest.mark.remote_data
def test_eleanor_lite():
    """Can we read in GSFC-ELEANOR-LITE light curves?"""
    url =
    "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/gsfc-eleanor-lite/s0001/0000/"
    "0003/3673/2616/hlsp_gsfc-eleanor-lite_tess_ffi_s0001-0000000336732616_tess_v1.0_lc.fits"
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a GSFC-ELEANOR-LITE file?
        assert detect_filetype(hdulist) == "GSFC-ELEANOR-LITE"
        # Are the correct fluxes read in?
        lc = read_eleanorlite_lightcurve(url, quality_bitmask=0)
        assert lc.meta["FLUX_ORIGIN"] == "corr_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["CORR_FLUX"])
        # Are the correct quality flags read in?
        lc = read_eleanorlite_lightcurve(url, quality_bitmask='default')
        assert ((lc["quality"] & 2**17) != 0).any() and ((lc["quality"] & 2**18) != 0).any()
        lc = read_eleanorlite_lightcurve(url, quality_bitmask='hardest')
        assert not (lc["quality"] & (2**17 | 2**18)).any()

@pytest.mark.remote_data
def test_search_eleanorlite():
    """Can we search and download GSFC-ELEANOR-LITE light curves from MAST?"""
    search = search_lightcurve("TIC 336732616", author="GSFC-ELEANOR-LITE", sector=1)
    assert len(search) == 1
    assert search.table["author"][0] == "GSFC-ELEANOR-LITE"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == 1
    assert lc.author == "GSFC-ELEANOR-LITE"
