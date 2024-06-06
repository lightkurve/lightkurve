"""Provides catalog data for `tpf.interact_sky()` function"""

from .core import *

from . import (
    gaia_tic,
    vsx,
    ztf,
)


def resolve_catalog_provider_class(name):
    if name == "gaiadr3":
        return gaia_tic.GaiaDR3InteractSkyCatalogProvider
    elif name == "gaiadr3_tic":
        return gaia_tic.GaiaDR3TICInteractSkyCatalogProvider
    elif name == "ztf":
        return ztf.ZTFInteractSkyCatalogProvider
    elif name == "vsx":
        return vsx.VSXInteractSkyCatalogProvider
    else:
        raise ValueError(f"Unsupported catalog: {name}")
