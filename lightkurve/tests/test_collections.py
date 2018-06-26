import pytest
import numpy as np

from ..lightcurve import (LightCurve, KeplerLightCurve, TessLightCurve,
                          iterative_box_period_search, LightCurveCollection)
from ..lightcurvecollection import LightCurveCollection
from ..lightcurvefile import KeplerLightCurveFile, TessLightCurveFile
import sys
sys.path.append("..") 

# 8th Quarter of Tabby's star
TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")
K2_C08 = ("https://archive.stsci.edu/missions/k2/lightcurves/c8/"
          "220100000/39000/ktwo220139473-c08_llc.fits")
KEPLER10 = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
            "0119/011904151/kplr011904151-2010009091648_llc.fits")
TESS_SIM = ("https://archive.stsci.edu/missions/tess/ete-6/tid/00/000/"
            "004/104/tess2019128220341-0000000410458113-0016-s_lc.fits")


def test_collections_plot():
	lc_1 = lightcurve.LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
	lc_2 = lightcurve.LightCurve(time=np.arange(1, 5), flux=np.arange(6, 10), flux_err=np.arange(1, 5))
	collection = LightCurveCollection((lc_1,lc_2))
	collection.plot()

