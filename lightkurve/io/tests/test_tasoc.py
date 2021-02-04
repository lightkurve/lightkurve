import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from ... import search_lightcurve
from ..tasoc import read_tasoc_lightcurve
from ..detect import detect_filetype

@pytest.mark.remote_data
def test_detect_tasoc():
    """Can we detect the correct format for TASOC files?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tasoc/s0001/ffi/0000/0004/1206/4070/hlsp_tasoc_tess_ffi_tic00412064070-s01-c1800_tess_v04_lc.fits"
    f = fits.open(url)

    assert detect_filetype(f)=="TASOC"


@pytest.mark.remote_data
def test_read_tasoc():
    """Can we read TASOC files?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/tasoc/s0001/ffi/0000/0004/1206/4070/hlsp_tasoc_tess_ffi_tic00412064070-s01-c1800_tess_v04_lc.fits"
    with fits.open(url, mode="readonly") as hdulist:
        fluxes = hdulist[1].data['FLUX_RAW']
        
    lc = read_tasoc_lightcurve(url, flux_column='FLUX_RAW')

    flux_lc = lc.flux.value

    #print(flux_lc, fluxes)
    assert np.sum(fluxes) == np.sum(flux_lc)

@pytest.mark.remote_data
def test_search_tasoc():
    """Can we search and download a TASOC light curve?"""
    search = search_lightcurve("TIC 412064070", author="TASOC", sector=1)
    assert len(search) == 1
    assert search.table["author"][0] == "TASOC"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == 1
