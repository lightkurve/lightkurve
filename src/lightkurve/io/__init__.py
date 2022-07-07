"""The .io sub-package provides functions for reading data."""
from .detect import *
from .read import *

from . import kepler, tess, qlp, k2sff, everest, pathos, tasoc, kepseismic, eleanorlite
from . import cdips
from .. import LightCurve

from astropy.io import registry

__all__ = ["read", "open"]


# TODO: to be replaced by register_data_product call below
# We intend the reader functions to be accessed via `LightCurve.read()`,
# so we add them to the `astropy.io.registry`.
try:
    registry.register_reader("kepler", LightCurve, kepler.read_kepler_lightcurve)
    registry.register_reader("tess", LightCurve, tess.read_tess_lightcurve)
    registry.register_reader("qlp", LightCurve, qlp.read_qlp_lightcurve)
    registry.register_reader("eleanorlite", LightCurve, eleanorlite.read_eleanorlite_lightcurve)
    registry.register_reader("k2sff", LightCurve, k2sff.read_k2sff_lightcurve)
    registry.register_reader("everest", LightCurve, everest.read_everest_lightcurve)
    registry.register_reader("pathos", LightCurve, pathos.read_pathos_lightcurve)
    registry.register_reader('cdips', LightCurve, cdips.read_cdips_lightcurve)
    registry.register_reader("tasoc", LightCurve, tasoc.read_tasoc_lightcurve)
    registry.register_reader("kepseismic", LightCurve, kepseismic.read_kepseismic_lightcurve)
except registry.IORegistryError:
    pass  # necessary to enable autoreload during debugging


AUTHOR_LINKS = {}  # to be used by search.py
DETECTORS = []  # to be used by detect.py
READER_SPECS = {}  # to be used by read.py


def register_data_product(module):
    spec = module.READER_SPEC
    READER_SPECS[spec[0]] = spec
    try:
        registry.register_reader(spec[0], spec[1], spec[2])
    except registry.IORegistryError:
        pass  # necessary to enable autoreload during debugging
    DETECTORS.append(module.detect_filetype)
    AUTHOR_LINKS.update(module.AUTHOR_LINKS)


#
# Register pre-shipped data products
#
register_data_product(qlp)
