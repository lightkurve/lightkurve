"""The .io sub-package provides functions for reading data."""
from .detect import *
from .read import *

from . import kepler, tess, qlp, k2sff, everest
from .. import LightCurve

from astropy.io import registry


__all__ = ['read', 'open']


# We intend the reader functions to be accessed via `LightCurve.read()`,
# so we add them to the `astropy.io.registry`.
try:
    registry.register_reader('kepler', LightCurve, kepler.read_kepler_lightcurve)
    registry.register_reader('tess', LightCurve, tess.read_tess_lightcurve)
    registry.register_reader('qlp', LightCurve, qlp.read_qlp_lightcurve)
    registry.register_reader('k2sff', LightCurve, k2sff.read_k2sff_lightcurve)
    registry.register_reader('everest', LightCurve, everest.read_everest_lightcurve)
except registry.IORegistryError:
    pass  # necessary to enable autoreload during debugging
