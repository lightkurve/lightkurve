import pytest

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.eleanor import read_eleanor_lightcurve
from lightkurve.io.detect import detect_filetype


@pytest.mark.remote_data
def test_gsfc_eleanor_lite():
    """Can we read in GSFC-ELEANOR-LITE light curves?"""
    url = (
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/gsfc-eleanor-lite/s0001/0000/"
        "0003/3673/2616/hlsp_gsfc-eleanor-lite_tess_ffi_s0001-0000000336732616_tess_v1.0_lc.fits"
    )
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a GSFC-ELEANOR-LITE file?
        assert detect_filetype(hdulist) == "ELEANOR"
        # Are the correct fluxes read in?
        lc = read_eleanor_lightcurve(url, quality_bitmask=0)
        assert lc.meta["AUTHOR"] == "GSFC-ELEANOR-LITE"
        assert lc.meta["FLUX_ORIGIN"] == "corr_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["CORR_FLUX"])
        # Are the correct quality flags read in?
        lc = read_eleanor_lightcurve(url, quality_bitmask='default')
        assert ((lc["quality"] & 2**17) != 0).any() and ((lc["quality"] & 2**18) != 0).any()
        lc = read_eleanor_lightcurve(url, quality_bitmask='hardest')
        assert not (lc["quality"] & (2**17 | 2**18)).any()
        assert np.issubdtype(lc["cadenceno"].dtype, np.integer)


@pytest.mark.parametrize(
    "url",
    [
        get_pkg_data_filename("../data/test-lc-tess-pimen_s1_eleanor_lite-100-cadences.fits"),
        # full version can also be read, though the full-specific data is not explicitly handled
        get_pkg_data_filename("../data/test-lc-tess-pimen_s1_eleanor_full-100-cadences.fits"),
    ],
)
def test_vanilla_eleanor(url):
    """Can we read in vanilla eleanor light curves?"""
    with fits.open(url, mode="readonly") as hdulist:
        # Can we auto-detect a vanilla eleanor file?
        assert detect_filetype(hdulist) == "ELEANOR"
        # Are the correct fluxes read in?
        lc = read_eleanor_lightcurve(url, quality_bitmask=0)
        assert lc.meta["AUTHOR"] == "ELEANOR"
        assert lc.meta["FLUX_ORIGIN"] == "corr_flux"
        assert_array_equal(lc.flux.value, hdulist[1].data["CORR_FLUX"])

        # vanilla eleanor can also contain PSF flux, ensure it is read in
        assert_array_equal(lc.psf_flux.value, hdulist[1].data["PSF_FLUX"])

        # vanilla eleanor's output cadenceno (FFIINDEX) dtype in the
        # FITS file is float, breaking convention
        # ensure we compensate it.
        assert np.issubdtype(lc["cadenceno"].dtype, np.integer)


@pytest.mark.remote_data
def test_search_gsfc_eleanor_lite():
    """Can we search and download GSFC-ELEANOR-LITE light curves from MAST?"""
    search = search_lightcurve("TIC 336732616", author="GSFC-ELEANOR-LITE", sector=1)
    assert len(search) == 1
    assert search.table["author"][0] == "GSFC-ELEANOR-LITE"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == 1
    assert lc.author == "GSFC-ELEANOR-LITE"
