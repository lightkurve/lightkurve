import pytest
from ..lightcurvefile import KeplerLightCurveFile

# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")


@pytest.mark.remote_data
def test_lightcurve_seismology_plot():
    """Sanity check to verify that periodogram plotting works"""
    KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX.periodogram().plot()
