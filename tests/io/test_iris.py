import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve.io.iris import read_iris_lightcurve
from lightkurve import search_lightcurve


@pytest.mark.remote_data
def test_read_iris():
    """Can we read IRIS files?"""
    url = "http://archive.stsci.edu/hlsps/iris/002400000/002437745/hlsp_iris_kepler_kepler_kplr002437745-stitched_kepler_v1.0_lc.fits"
    f = fits.open(url)
    # verify can open
    lc = read_iris_lightcurve(url)
    assert type(lc).__name__ == "KeplerLightCurve"
    # Are `time` and `flux` consistent with the FITS file?
    assert_array_equal(f[1].data["TIME"], lc.time.value)
    assert_array_equal(f[1].data["CORRECTED_FLUX"] + 1, lc.flux.value)


@pytest.mark.remote_data
def test_search_iris():
    """Can we search and download an IRIS light curve?"""
    # Try an early campaign
    search = search_lightcurve("KIC 2437745", author="IRIS")
    assert len(search) == 18
    assert search.table["author"][0] == "IRIS"
    lc = search.download()
    assert type(lc).__name__ == "KeplerLightCurve"
    assert np.isclose(np.median(lc.flux.value), 1, atol=0.001)
