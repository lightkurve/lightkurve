"""Astropy units has percent, but no ppt or ppm."""
import astropy.units as u

__all__ = []

ppt = u.def_unit(['ppt', 'parts per thousand'], u.Unit(1e-3))
ppm = u.def_unit(['ppm', 'parts per million'], u.Unit(1e-6))
u.add_enabled_units([ppt, ppm])
