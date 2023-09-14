import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve import search_lightcurve, LightCurveCollection
from lightkurve.io.cdips import read_cdips_lightcurve
from lightkurve.io.detect import detect_filetype

TEST_TIC_ID = 104669918
TEST_FIT_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/cdips/s0014/cam1_ccd1/hlsp_cdips_tess_ffi_gaiatwo0002030897830235411200-s0014-cam1-ccd1_tess_v01_llc.fits"


@pytest.mark.remote_data
def test_detect_cdips():
    """Can we detect the correct format for CDIPS files?"""
    url = TEST_FIT_URL
    f = fits.open(url)

    assert detect_filetype(f) == "CDIPS"


@pytest.mark.remote_data
def test_read_cdips():
    """Can we read CDIPS files?"""
    url = TEST_FIT_URL
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
        assert not np.array_equal(fluxes[i].value, fluxes[i+1].value)


@pytest.mark.remote_data
def test_search_cdips():
    """Can we search and download a cdips light curve?"""
    search = search_lightcurve(f"TIC {TEST_TIC_ID}", author="CDIPS")
    assert len(search) >= 1
    assert search.table["author"][0] == "CDIPS"
    lc = search.download()
    assert type(lc).__name__ == "TessLightCurve"
    assert hasattr(lc, "sector")
    assert str(lc['bge'].unit) == 'adu'
    slc = LightCurveCollection([lc, lc]).stitch()
    assert len(slc) == 2*len(lc)

