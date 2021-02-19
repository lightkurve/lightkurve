import pytest

import lightkurve as lk  # necessary to enable the units tested below
from astropy import units as u


def test_custom_units():
    """Are ppt, ppm, and percent enabled AstroPy units?"""
    u.Unit("ppt")     # custom unit defined in lightkurve.units
    u.Unit("ppm")     # not enabled by default; enabled in lightkurve.units
    u.Unit("percent") # standard AstroPy unit


@pytest.mark.remote_data
def test_tasoc_ppm_units():
    """Regression test for #956."""
    lc = lk.search_lightcurve('HV 2112', author='TASOC', sector=1, exptime=1800).download()
    assert lc['flux_corr'].unit == "ppm"
    assert "Unrecognized" not in repr(lc['flux_corr'].unit)
