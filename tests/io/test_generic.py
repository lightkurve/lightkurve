import pytest
from lightkurve.io import read
from astropy.io import fits
from astropy import units as u
import numpy as np
from astropy.table import Table
from lightkurve.utils import LightkurveError
import os
from .. import TESTDATA

@pytest.mark.filterwarnings("error::ResourceWarning")
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore::astropy.units.UnitsWarning")  # Suppress astropy complaint about slashes in unit strings
def test_read_generic():
    """
    Can we read generic light curve files from other missions?
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
        hdu0 = hdul[0].copy()  # for use in permutations

    # Is the time axis calculated correctly?
    assert all(lc.time.value == (data['TIME'] + header1['JDREF']))
    
    # Is time format correctly inferred as JD?
    assert lc.time.format.lower() == "jd"

    # Is the time scale correctly read in as TDB?
    assert lc.time.scale.lower() == header1['TIMESYS'].lower()
    
    # Are other data and metadata correctly read in?
    assert all((np.ma.asarray(lc.flux.value) == np.ma.asarray(data['FLUX'])).compressed())
    assert all((np.ma.asarray(lc.ctr_err.value) == np.ma.asarray(data['CTR_ERR'])).compressed())
    assert lc.label == header0['OBJECT']
    assert len(lc) == len(data)
    assert lc.flux.unit == u.Unit(header1['TUNIT5'])

    # ------- Start checking other permutations

    # ------- PERMUTATION 1
    # MJDREF, erroneous lone MJDREFF, erroneous JDREF 
    new_header1 = header1.copy()
    new_header1["MJDREF"] = 300.  # Use MJDREF
    new_header1["JDREF"] = 2400000.5  # Incorrect vestigial JDREF to ignore
    new_header1["MJDREFF"] = 0.5  # Incorrect MJDREFF to ignore without a corresponding MJDREFI

    # Create new lc object
    new_hdu1 = fits.BinTableHDU(data=data, header=new_header1)
    new_hdulist = fits.HDUList(hdus=[hdu0, new_hdu1])
    new_lc = read(new_hdulist)

    # Is the time axis calculated correctly?
    assert all(new_lc.time.value == (data['TIME'] + new_header1['MJDREF']))
    # Is time format correctly inferred as MJD?
    assert new_lc.time.format.lower() == "mjd"

    # ------- PERMUTATION 2
    # JDREFI + JDREFF, erroneous JDREF, erroneous lone MJDREFF
    new_header1 = header1.copy()
    new_header1["JDREFI"] = 2400009  # Use JDREFI + JDREFF
    new_header1["JDREFF"] = 0.1
    new_header1["JDREF"] = 2400000.5  # Incorrect vestigial JDREF to ignore
    new_header1["MJDREFF"] = 0.5  # Incorrect MJDREFF to ignore without a corresponding MJDREFI

    # Create new lc object
    new_hdu1 = fits.BinTableHDU(data=data, header=new_header1)
    new_hdulist = fits.HDUList(hdus=[hdu0, new_hdu1])
    new_lc = read(new_hdulist)

    # Is the time axis calculated correctly?
    assert all(new_lc.time.value == (data['TIME'] + new_header1['JDREFI'] + new_header1['JDREFF']))
    # Is time format correctly inferred as JD?
    assert new_lc.time.format.lower() == "jd"

    # ------- PERMUTATION 3
    # Not enough information to compute time array
    new_header1 = header1.copy()
    del new_header1["JDREF"]
    new_header1["JDREFI"] = 2400009  # Incorrect JDREFI to ignore without a corresponding JDREFF
    new_header1["MJDREFF"] = 0.5  # Incorrect MJDREFF to ignore without a corresponding MJDREFI

    # Prep to create new lc object
    new_hdu1 = fits.BinTableHDU(data=data, header=new_header1)
    new_hdulist = fits.HDUList(hdus=[hdu0, new_hdu1])

    # Lightkurve should fail to read
    with pytest.raises(LightkurveError) as excinfo:
        _ = read(new_hdulist)

    # Check that underlying exception is ValueError
    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    
    # ------- PERMUTATION 4
    # MJDREFI+MJDREFF, erroneous JDREF, different time scale, time column in seconds
    new_header1 = header1.copy()
    new_header1["MJDREFI"] = 1234  # Use MJDREFI + MJDREFF
    new_header1["MJDREFF"] = 0.123456789  # Use MJDREFI + MJDREFF
    new_header1["JDREF"] = 2400000.5  # Incorrect vestigial JDREF to test that it is ignored
    new_header1["TUNIT1"] = 's'  # Test unit change
    new_header1["TIMESYS"] = 'UTC'  # Test time scale read in
    new_header1["TCTYP1"] = 'UTC'  # consistency for future-proofing this test
    new_header1["TREFPOS"] = 'TOPOCENTER'  # consistency for future-proofing this test
    new_header1["TRPOS1"] = 'TOPOCENTER'  # consistency for future-proofing this test

    # Convert units in `data`, otherwise astropy will override the header units
    datatable = Table(data)
    datatable["TIME"].unit = u.s

    # Create new lc object
    new_hdu1 = fits.BinTableHDU(data=datatable, header=new_header1)
    new_hdulist = fits.HDUList(hdus=[hdu0, new_hdu1])
    new_lc = read(new_hdulist)

    # Is the time axis calculated correctly, with unit conversion
    # and using the right reference time (MJDREFI+MJDREFF, not JDREF)?
    time_seconds = data['TIME']*u.Unit(new_header1['TUNIT1'])
    time_days = time_seconds.to_value('d')
    assert all(new_lc.time.value == (time_days + new_header1['MJDREFI'] + new_header1['MJDREFF']))

    # Is time format correctly inferred as MJD?
    assert new_lc.time.format.lower() == "mjd"

    # Is the time scale correctly read in as UTC?
    assert new_lc.time.scale.lower() == new_header1['TIMESYS'].lower()
