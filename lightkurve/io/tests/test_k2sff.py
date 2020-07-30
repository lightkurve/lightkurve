import pytest

from astropy.io import fits
from numpy.testing import assert_array_equal

from ..k2sff import read_k2sff_lightcurve


@pytest.mark.remote_data
def test_read_k2sff():
    """Can we read K2SFF files?"""
    url = "http://archive.stsci.edu/hlsps/k2sff/c16/212100000/00236/hlsp_k2sff_k2_lightcurve_212100236-c16_kepler_v1_llc.fits"
    f = fits.open(url)
    # Verify different extensions
    for ext in ["BESTAPER", "CIRC_APER9"]:
        lc = read_k2sff_lightcurve(url, ext=ext)
        assert type(lc).__name__ == "KeplerLightCurve"
        # Are `time` and `flux` consistent with the FITS file?
        assert_array_equal(f[ext].data['T'], lc.time.value)
        assert_array_equal(f[ext].data['FCOR'], lc.flux.value)
