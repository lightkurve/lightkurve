import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.qlp import read_qlp_lightcurve
from lightkurve.io.detect import detect_filetype


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "url, flux_err_colname_expected, qlp_low_precision_bitmask", [
        ("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/qlp/s0011/0000/0002/7755/4109/hlsp_qlp_tess_ffi_s0011-0000000277554109_tess_v01_llc.fits",
         "KSPSAP_FLUX_ERR",  # for sectors 1 - 55
         2**12,  # bit 13 for sectors 1 -55
         ),
        ("https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/qlp/s0056/0000/0000/1054/9159/hlsp_qlp_tess_ffi_s0056-0000000010549159_tess_v01_llc.fits",
         "DET_FLUX_ERR",  # for sectors 56+
         2**30,  # bit 31 for sectors 56+
         ),
        ])
def test_qlp(url, flux_err_colname_expected, qlp_low_precision_bitmask):
    """Can we read in QLP light curves?"""
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a QLP file?
        assert detect_filetype(hdulist) == "QLP"
        # Are the correct fluxes read in?
        lc = read_qlp_lightcurve(url, quality_bitmask=0)
        assert lc.meta["FLUX_ORIGIN"] == "sap_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["SAP_FLUX"])
        assert_array_equal(lc.flux_err.value, hdulist[1].data[flux_err_colname_expected])

        # Test handling of QLP-specific low-precision bitmask
        # - the cadences marked as such will be masked out by "hard" / "hardest"

        # first assure the test FITS file has cadence marked by QLP bit only
        # to easily isolate the effects of the quality_bitmask
        assert (lc["quality"] == qlp_low_precision_bitmask).any()

        lc = read_qlp_lightcurve(url, quality_bitmask="default")
        assert (lc["quality"] & (qlp_low_precision_bitmask)).any()

        lc = read_qlp_lightcurve(url, quality_bitmask="hard")
        assert not (lc["quality"] & (qlp_low_precision_bitmask)).any()

        lc = read_qlp_lightcurve(url, quality_bitmask="hardest")
        assert not (lc["quality"] & (qlp_low_precision_bitmask)).any()


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
