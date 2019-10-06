import numpy as np
from numpy.testing import assert_almost_equal

from ... import LightCurve
from .. import SFFCorrector

K2_C08 = ("https://archive.stsci.edu/missions/k2/lightcurves/c8/"
          "220100000/39000/ktwo220139473-c08_llc.fits")


@pytest.mark.remote_data
@pytest.mark.parametrize("path", K2_C08)])
def test_remote_data():
    """Can we correct a simple K2 light curve?"""

    lcf = KeplerLightCurveFile(path, quality_bitmask=None)
    sff = SFFCorrector(lcf.PDCSAP_FLUX, windows=10, bins=5, timescale=0.5)
    corrected_lc = sff.correct()
