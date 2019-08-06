import matplotlib.pyplot as pl

from ... import search_lightcurvefile
from .. import GPCorrector

def test_gpcorrector():
    lc = search_lightcurvefile("Kepler-10", quarter=4).download().PDCSAP_FLUX
    gpc = GPCorrector(lc, kernel="matern32")
    gpc.diagnose()
    pl.savefig("/tmp/gp-matern.png")
    gpc.optimize()
    gpc.diagnose()
    pl.savefig("/tmp/gp-matern-optimized.png")
    gpc.correct()

    gpc = GPCorrector(lc, kernel="sho")
    gpc.diagnose()
    pl.savefig("/tmp/gp-sho.png")
    gpc.optimize()
    gpc.diagnose()
    pl.savefig("/tmp/gp-sho-optimized.png")
    gpc.correct()
