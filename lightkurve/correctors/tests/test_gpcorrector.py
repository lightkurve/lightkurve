import pytest

from ... import search_lightcurvefile
from .. import GPCorrector

@pytest.mark.remote_data
def test_gpcorrector():
    lc = search_lightcurvefile("Kepler-10", quarter=10).download().PDCSAP_FLUX
    # Try Matern
    gpc = GPCorrector(lc, kernel="matern32")
    gpc.optimize()
    gpc.diagnose()
    corr_lc = gpc.correct()
    assert corr_lc.estimate_cdpp() < lc.estimate_cdpp()

    # Try Sho
    gpc = GPCorrector(lc, kernel="sho")
    gpc.optimize()
    gpc.diagnose()
    corr_lc = gpc.correct()
    assert corr_lc.estimate_cdpp() < lc.estimate_cdpp()

    # Try from LC object
    corr_lc = lc.to_corrector("gp").correct()
    assert corr_lc.estimate_cdpp() < lc.estimate_cdpp()
