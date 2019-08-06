import pytest
import matplotlib.pyplot as pl

from ... import search_lightcurvefile
from .. import GPCorrector

@pytest.mark.remote_data
def test_gpcorrector():
    lc = search_lightcurvefile("Kepler-10", quarter=4).download().PDCSAP_FLUX
    # Try Matern
    gpc = GPCorrector(lc, kernel="matern32")
    gpc.diagnose()
    pl.savefig("/tmp/gp-matern.png")
    gpc.optimize()
    gpc.diagnose()
    pl.savefig("/tmp/gp-matern-optimized.png")
    gpc.correct()
    # Try Sho
    gpc = GPCorrector(lc, kernel="sho")
    gpc.diagnose()
    pl.savefig("/tmp/gp-sho.png")
    gpc.optimize()
    gpc.diagnose()
    pl.savefig("/tmp/gp-sho-optimized.png")
    gpc.correct()
    # Try from LC object
    corr_lc = lc.to_corrector("gp").correct()
    assert corr_lc.estimate_cdpp() < lc.estimate_cdpp()
