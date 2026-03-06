import pytest
from lightkurve.io import read
from astropy.io import fits
from astropy import units as u
import os
from .. import TESTDATA

@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore::astropy.units.UnitsWarning")  # Suppress astropy complaint about slashes in unit strings
def test_read_generic_jdref():
    """
    Can we read generic light curve files from other missions?
    Part 1: simple fake SPARCS light curve with JDREF.
    """

    # Open the file in lightkurve
    filename = os.path.join(TESTDATA, "test-sparcs-jdref.fits")
    lc = read(filename)
    assert type(lc).__name__ == "LightCurve"

    # Open the file in astropy.fits
    with fits.open(filename) as hdul:
        data = hdul[1].data
        header0 = hdul[0].header
        header1 = hdul[1].header

    # Is the time axis calculated correctly?
    assert lc.time[3].value == (data['TIME'][3] + header1['JDREF'])

    # Is time format correctly inferred as JD?
    assert lc.time.format.lower() == "jd"

    # Is the time scale correctly read in as TDB?
    assert lc.time.scale.lower() == header1['TIMESYS'].lower()
    
    # Are other data and metadata correctly read in?
    assert float(lc.flux[5].value) == data['FLUX'][5]
    assert float(lc.ctr_err[5].value) == data['CTR_ERR'][5]
    assert lc.label == header0['OBJECT']
    assert len(lc) == len(data)
    assert lc.flux.unit == u.Unit(header1['TUNIT5'])


@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore::astropy.units.UnitsWarning")  # Suppress astropy complaint about slashes in unit strings
def test_read_generic_mjdrefif():
    """
    Can we read generic light curve files from other missions?
    Part 2: tricky fake SPARCS light curve with MJDREFI+MJDREFF
    and a superfluous bad JDREF, different time scale, time column in seconds.
    """

    # Open the file in lightkurve
    filename = os.path.join(TESTDATA, "test-sparcs-mjdrefif-badjdref-unitchange.fits")
    lc = read(filename)
    assert type(lc).__name__ == "LightCurve"

    # Open the file in astropy.fits
    with fits.open(filename) as hdul:
        data = hdul[1].data
        header1 = hdul[1].header

    # Is the time axis calculated correctly, with unit conversion
    # and using the right reference time (MJDREFI+MJDREFF, not JDREF)?
    time_seconds = data['TIME'][2]*u.Unit(header1['TUNIT1'])
    time_days = time_seconds.to_value('d')
    assert lc.time[2].value == (time_days + header1['MJDREFI'] + header1['MJDREFF'])

    # Is time format correctly inferred as MJD?
    assert lc.time.format.lower() == "mjd"

    # Is the time scale correctly read in as UTC?
    assert lc.time.scale.lower() == header1['TIMESYS'].lower()
