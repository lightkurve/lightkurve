import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve.io.kbonus import read_kbonus_lightcurve, read_kbonus_lightcurve_quarters
from lightkurve import search_lightcurve


@pytest.mark.remote_data
def test_read_kbonus():
    """Can we read KBONUS files?"""
    url = "http://archive.stsci.edu/hlsps/kbonus-bkg/lcs/0119/011904151/hlsp_kbonus-bkg_kepler_kepler_kic-011904151_kepler_v1.0_lc.fits"
    f = fits.open(url)
    # verify can open
    lc = read_kbonus_lightcurve(url)
    assert type(lc).__name__ == "KeplerLightCurve"
    # Are `time` and `flux` consistent with the FITS file?
    assert_array_equal(f[1].data["TIME"], lc.time.value)
    assert_array_equal(f[1].data["FLUX"], lc.flux.value)
    assert lc.FLFRCSAP > 0.94
    assert lc.CROWDSAP == 1.0

    # In this special case we get a light curve collection
    lcs = read_kbonus_lightcurve_quarters(url, quarters=[2, 10])
    assert type(lcs).__name__ == "LightCurveCollection"
    assert len(lcs) == 2
    assert_array_equal(np.asarray([lc.quarter for lc in lcs]), np.asarray([2, 10]))


@pytest.mark.remote_data
def test_search_kbonus(caplog):
    """Can we search and download a KBONUS light curve?"""
    # Try an early campaign
    search = search_lightcurve("Kepler-10", author="KBONUS-BKG")
    assert len(search) == 1
    assert search.table["author"][0] == "KBONUS-BKG"
    lc = search.download()
    # Kepler-10 should have 2 warnings to user
    assert len(caplog.records) == 2
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert (
        "KBONUS-BKG data is invalid for saturated targets" in caplog.records[0].message
    )
    assert "This data may be unreliable." in caplog.records[1].message
    assert type(lc).__name__ == "KeplerLightCurve"
    lcs = search.download(quarters=[2, 10])
    assert type(lcs).__name__ == "LightCurveCollection"
    assert_array_equal(np.asarray([lc.quarter for lc in lcs]), np.asarray([2, 10]))
