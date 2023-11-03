"""The .io sub-package provides functions for reading data."""
from astropy.io import registry

from .. import LightCurve
from . import (
    cdips,
    eleanor,
    everest,
    k2sff,
    kepler,
    kepseismic,
    pathos,
    qlp,
    tasoc,
    tess,
    tglc,
    kbonus,
    iris,
)
from .detect import *
from .read import *

__all__ = ["read", "open"]


# Which external links should we display in the SearchResult repr?
AUTHOR_LINKS = {
    "Kepler": "https://archive.stsci.edu/kepler/data_products.html",
    "K2": "https://archive.stsci.edu/k2/data_products.html",
    "SPOC": "https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html",
    "TESS": "https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html",
    "TESS-SPOC": "https://archive.stsci.edu/hlsp/tess-spoc",
    "QLP": "https://archive.stsci.edu/hlsp/qlp",
    "TASOC": "https://archive.stsci.edu/hlsp/tasoc",
    "PATHOS": "https://archive.stsci.edu/hlsp/pathos",
    "CDIPS": "https://archive.stsci.edu/hlsp/cdips",
    "K2SFF": "https://archive.stsci.edu/hlsp/k2sff",
    "EVEREST": "https://archive.stsci.edu/hlsp/everest",
    "TESScut": "https://mast.stsci.edu/tesscut/",
    "GSFC-ELEANOR-LITE": "https://archive.stsci.edu/hlsp/gsfc-eleanor-lite",
    "TGLC": "https://archive.stsci.edu/hlsp/tglc",
    "KBONUS-BKG": "https://archive.stsci.edu/hlsp/kbonus-bkg",
    "KEPSEISMIC": "https://archive.stsci.edu/prepds/kepseismic/",
    "IRIS": "https://archive.stsci.edu/hlsp/iris",
}

# Note if a HLSP does not appear in this list it no longer appears in the MAST query.
# This is to prevent new HLSPs that do not conform to fits standards from breaking queries.
# To add a new HLSP, open a PR with a new reader.

# We intend the reader functions to be accessed via `LightCurve.read()`,
# so we add them to the `astropy.io.registry`.
try:
    registry.register_reader("kepler", LightCurve, kepler.read_kepler_lightcurve)
    registry.register_reader("tess", LightCurve, tess.read_tess_lightcurve)
    registry.register_reader("qlp", LightCurve, qlp.read_qlp_lightcurve)
    registry.register_reader("eleanor", LightCurve, eleanor.read_eleanor_lightcurve)
    registry.register_reader("k2sff", LightCurve, k2sff.read_k2sff_lightcurve)
    registry.register_reader("everest", LightCurve, everest.read_everest_lightcurve)
    registry.register_reader("pathos", LightCurve, pathos.read_pathos_lightcurve)
    registry.register_reader("cdips", LightCurve, cdips.read_cdips_lightcurve)
    registry.register_reader("tasoc", LightCurve, tasoc.read_tasoc_lightcurve)
    registry.register_reader(
        "kepseismic", LightCurve, kepseismic.read_kepseismic_lightcurve
    )
    registry.register_reader("iris", LightCurve, iris.read_iris_lightcurve)
    registry.register_reader("tglc", LightCurve, tglc.read_tglc_lightcurve)
    registry.register_reader("kbonus", LightCurve, kbonus.read_kbonus_lightcurve)
except registry.IORegistryError:
    pass  # necessary to enable autoreload during debugging
