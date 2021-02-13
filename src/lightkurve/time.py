"""Adds the BKJD and BTJD time format for use by Astropy's `Time` object."""
from astropy.time.formats import TimeNumeric, day_frac


class TimeBKJD(TimeNumeric):
    """
    Barycentric Kepler Julian Date time format.
    This represents the number of days since January 1, 2009 12:00:00 UTC.
    BKJD is the format in which times are recorded in Kepler data products.
    See Section 2.3.2 in the Kepler Archive Manual for details.
    """

    name = "bkjd"
    BKJDREF = 2454833  # Barycentric Kepler Julian Date offset

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        jd1, jd2 = day_frac(val1, val2)
        jd1 += self.BKJDREF
        self.jd1, self.jd2 = day_frac(jd1, jd2)

    def to_value(self, **kwargs):
        jd1 = self.jd1 - self.BKJDREF
        jd2 = self.jd2
        return super().to_value(jd1=jd1, jd2=jd2, **kwargs)

    value = property(to_value)


class TimeBTJD(TimeNumeric):
    """
    Barycentric TESS Julian Date time format.
    This represents the number of days since JD 2457000.0.
    BTJD is the format in which times are recorded in TESS data products.
    """

    name = "btjd"
    BTJDREF = 2457000  # Barycentric TESS Julian Date offset

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)  # Validate scale.
        jd1, jd2 = day_frac(val1, val2)
        jd1 += self.BTJDREF
        self.jd1, self.jd2 = day_frac(jd1, jd2)

    def to_value(self, **kwargs):
        jd1 = self.jd1 - self.BTJDREF
        jd2 = self.jd2
        return super().to_value(jd1=jd1, jd2=jd2, **kwargs)

    value = property(to_value)
