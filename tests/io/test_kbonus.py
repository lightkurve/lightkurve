import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve.io.kbonus import read_kbonus_lightcurve
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

    lc = read_kbonus_lightcurve(url, quarter=0)
    assert type(lc).__name__ == "KeplerLightCurve"
    assert lc.FLFRCSAP > 0.94
    assert lc.CROWDSAP == 1.0


@pytest.mark.remote_data
def test_search_kbonus(caplog):
    """Can we search and download a KBONUS light curve?"""
    # Try an early campaign
    search = search_lightcurve("Kepler-10", author="KBONUS-BKG")
    assert len(search) == 19
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
    lcs = search[1].download()
    assert type(lcs).__name__ == "KeplerLightCurve"
    lcs = search[1:3].download_all()
    assert type(lcs).__name__ == "LightCurveCollection"
    assert_array_equal(np.asarray([lc.quarter for lc in lcs]), np.asarray([0, 1]))
