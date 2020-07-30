from .detect import *
from .read import *

from . import official_products, k2sff
from .. import LightCurve

from astropy.io import registry

"""ADD READERS TO THE REGISTRY"""
try:
    registry.register_reader('kepler', LightCurve, official_products.read_kepler_lightcurve)
    registry.register_reader('tess', LightCurve, official_products.read_tess_lightcurve)
    registry.register_reader('k2sff', LightCurve, k2sff.read_k2sff_lightcurve)
except registry.IORegistryError:
    pass  # necessary to enable autoreload during debugging
