import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from ..k2sff import read_k2sff_lightcurve
from ... import search_lightcurve


@pytest.mark.remote_data
def test_read_k2sff():
    """Can we read K2SFF files?"""
    url = "http://archive.stsci.edu/hlsps/k2sff/c16/212100000/00236/hlsp_k2sff_k2_lightcurve_212100236-c16_kepler_v1_llc.fits"
    f = fits.open(url)
    # Verify different extensions
    fluxes = []
    for ext in ["BESTAPER", "CIRC_APER9"]:
        lc = read_k2sff_lightcurve(url, ext=ext)
        assert type(lc).__name__ == "KeplerLightCurve"
        # Are `time` and `flux` consistent with the FITS file?
        assert_array_equal(f[ext].data['T'], lc.time.value)
        assert_array_equal(f[ext].data['FCOR'], lc.flux.value)
        fluxes.append(lc.flux)
    # Different extensions should show different fluxes
    assert not np.array_equal(fluxes[0] , fluxes[1])


@pytest.mark.remote_data
def test_search_k2sff():
    """Can we search and download a K2SFF light curve?"""
    # Try an early campaign
    search = search_lightcurve("K2-18", author="K2SFF", campaign=1)
    assert len(search) == 1
    assert search.table["author"][0] == "K2SFF"
    lc = search.download()
    assert type(lc).__name__ == "KeplerLightCurve"
    assert lc.campaign == 1
    # Try a late campaign
    lc = search_lightcurve("GJ 9827", author="K2SFF", campaign=19).download()
    assert type(lc).__name__ == "KeplerLightCurve"
    assert lc.targetid == 246389858
    assert lc.campaign == 19
