import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from ... import search_lightcurve
from ..pathos import read_pathos_lightcurve


@pytest.mark.remote_data
def test_read_pathos():
    """Can we read PATHOS files?"""
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/pathos/s0008/hlsp_pathos_tess_lightcurve_tic-0093270923-s0008_tess_v1_llc.fits"
    f = fits.open(url)
    # Verify different extensions
    fluxes = []

    exts = ['PSF_FLUX_RAW', 'PSF_FLUX_COR']
    exts.extend([f'AP{ap}_FLUX_RAW' for ap in [1,2,3,4]])
    exts.extend([f'AP{ap}_FLUX_COR' for ap in [1,2,3,4]])

    for ext in exts:
        lc = read_pathos_lightcurve(url, ext=ext)
        assert type(lc).__name__ == "TESSLightCurve"
        # Are `time` and `flux` consistent with the FITS file?
        assert_array_equal(f[1].data['TIME'], lc.time.value)
        assert_array_equal(f[1].data[ext], lc.flux.value)
        fluxes.append(lc.flux)
    # Different extensions should show different fluxes
    for i in range(9):
        assert not np.array_equal(fluxes[i] , fluxes[i+1])

@pytest.mark.remote_data
def test_search_pathos():
    """Can we search and download a PATHOS light curve?"""
    search = search_lightcurve("TIC 93270923", author="PATHOS", sector=8)
    assert len(search) == 1
    assert search.table["author"][0] == "PATHOS"
    lc = search.download()
    assert type(lc).__name__ == "TESSLightCurve"
    assert lc.sector == 8
