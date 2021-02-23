"""Define custom AstroPy units commonly used by the Kepler/TESS community."""
from astropy import units as u

__all__ = ["ppt", "ppm"]

ppt = u.def_unit(["ppt", "parts per thousand"], u.Unit(1e-3))
ppm = u.def_unit(["ppm", "parts per million"], u.Unit(1e-6))
u.add_enabled_units([ppt, ppm])
