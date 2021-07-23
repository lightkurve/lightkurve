"""Adds the BKJD and BTJD time format for use by Astropy's `Time` object.

Caution: AstroPy time objects make a distinction between a time's format
(e.g. ISO, JD, MJD) and its scale (e.g. UTC, TDB).  This can be confusing
because the acronym "BTJD" refers both to a format (TJD) and to a scale (TDB).

Note: the classes below derive from an AstroPy meta class which will automatically
register the formats for use in AstroPy Time objects.
"""
from astropy.time.formats import TimeFromEpoch


class TimeBKJD(TimeFromEpoch):
    """
    Barycentric Kepler Julian Date (BKJD): days since JD 2454833.0.

    For example, 0 in BTJD is noon on January 1, 2009.

    BKJD is the format in which times are recorded in data products from
    NASA's Kepler Space Telescope, where it is always given in the
    Barycentric Dynamical Time (TDB) scale by convention.
    """
    name = 'bkjd'
    unit = 1.0
    epoch_val = 2454833
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'


class TimeBTJD(TimeFromEpoch):
    """
    Barycentric TESS Julian Date (BTJD): days since JD 2457000.0.

    For example, 0 in BTJD is noon on December 8, 2014.

    BTJD is the format in which times are recorded in data products from
    NASA's Transiting Exoplanet Survey Satellite (TESS), where it is
    always given in the Barycentric Dynamical Time (TDB) scale by convention.
    """
    name = 'btjd'
    unit = 1.0
    epoch_val = 2457000
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'jd'
