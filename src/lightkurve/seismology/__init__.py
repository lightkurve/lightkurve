"""The `lightkurve.seismology` sub-package provides classes and functions for
quick-look asteroseismic analyses."""

# Do not export the modules in this subpackage to the root namespace, important
# because `lightkurve.utils` collides with `lightkurve.seismology.utils`.
__all__ = [
    "Seismology",
    "SeismologyQuantity",
    "estimate_numax_acf2d",
    "diagnose_numax_acf2d",
    "estimate_deltanu_acf2d",
    "diagnose_deltanu_acf2d",
    "estimate_radius",
    "estimate_mass",
    "estimate_logg",
]

from .core import Seismology
from .deltanu_estimators import diagnose_deltanu_acf2d, estimate_deltanu_acf2d
from .numax_estimators import diagnose_numax_acf2d, estimate_numax_acf2d
from .stellar_estimators import estimate_logg, estimate_mass, estimate_radius
from .utils import SeismologyQuantity
