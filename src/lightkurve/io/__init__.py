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
)
from .detect import *
from .read import *

__all__ = ["read", "open"]


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
    registry.register_reader("tglc", LightCurve, tglc.read_tglc_lightcurve)
except registry.IORegistryError:
    pass  # necessary to enable autoreload during debugging
