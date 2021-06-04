import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve
from lightkurve.io.cdips import read_cdips_lightcurve
from lightkurve.io.detect import detect_filetype

@pytest.mark.remote_data
def test_detect_cdips():
    """Can we detect the correct format for CDIPS files?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/cdips/s0008/cam3_ccd4/hlsp_cdips_tess_ffi_gaiatwo0005318059532750974720-0008-cam3-ccd4_tess_v01_llc.fits"
    f = fits.open(url)

    assert detect_filetype(f)=="CDIPS"


@pytest.mark.remote_data
def test_read_cdips():
    """Can we read CDIPS files?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/cdips/s0008/cam3_ccd4/hlsp_cdips_tess_ffi_gaiatwo0005318059532750974720-0008-cam3-ccd4_tess_v01_llc.fits"
    f = fits.open(url)
    # Verify different extensions
    fluxes = []

    # Test instrumental flux and magnitude, and detrended magnitudes
    exts = [f'IFL{ap}' for ap in [1,2,3]]
    exts.extend([f'IRM{ap}' for ap in [1,2,3]])
    exts.extend([f'TFA{ap}' for ap in [1,2,3]])
    exts.extend([f'PCA{ap}' for ap in [1,2,3]])

    for ext in exts:
        lc = read_cdips_lightcurve(url, flux_column=ext)
        assert type(lc).__name__ == "TessLightCurve"
        assert lc.meta["FLUX_ORIGIN"] == ext.lower()
        # Are `time` and `flux` consistent with the FITS file?
        assert_array_equal(f[1].data['TMID_BJD'][lc.meta['QUALITY_MASK']],
                           lc.time.value)
        assert_array_equal(f[1].data[ext][lc.meta['QUALITY_MASK']],
                           lc.flux.value)
        fluxes.append(lc.flux)
    # Different extensions should show different fluxes
    for i in range(11):
        assert not np.array_equal(fluxes[i].value , fluxes[i+1].value)

@pytest.mark.remote_data
def test_search_cdips():
    """Can we search and download a cdips light curve?"""
    search = search_lightcurve("TIC 93270923", author="CDIPS", sector=8)
    assert len(search) == 1
    assert search.table["author"][0] == "CDIPS"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert lc.sector == 8
