from astropy.time import Time
import numpy as np


def test_bkjd():
    """Tests for the Barycentric Kepler Julian Date (BKJD) time format."""
    # Sanity checks
    t0 = Time(0, format="bkjd")
    assert t0.format == "bkjd"
    assert t0.scale == "tdb"
    assert t0.iso == "2009-01-01 12:00:00.000"


def test_btjd():
    """Tests for the Barycentric TESS Julian Date (BTJD) time format."""
    # Sanity checks
    t0 = Time(0, format="btjd")
    assert t0.format == "btjd"
    assert t0.scale == "tdb"
    assert t0.iso == "2014-12-08 12:00:00.000"

    # The test values below correspond to the header keywords (TSTART, TSTOP, DATE-OBS, DATE-END)
    # found in s3://stpubdata/tess/public/ffi/s0031/2020/296/4-3/tess2020296001912-s0031-4-3-0198-s_ffic.fits
    tstart, tstop = 2144.513656838462, 2144.520601048349
    date_obs, date_end = '2020-10-22 00:18:30.767', '2020-10-22 00:28:30.747'
    assert np.isclose(Time(date_obs).btjd, tstart, rtol=1e-10)
    assert np.isclose(Time(date_end).btjd, tstop, rtol=1e-10)
    assert np.isclose(Time(date_end).btjd, Time(date_end).tdb.btjd, rtol=1e-10)
    assert Time(tstart, format="btjd").utc.iso[:22] == date_obs[:22]
    assert Time(tstop, format="btjd").utc.iso[:22] == date_end[:22]
