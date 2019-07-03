"""Defines the asteroseismology module"""
from __future__ import division, print_function

import copy
import os
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt
import astropy
from astropy import units as u
from astropy import constants as const

from uncertainties import ufloat
from uncertainties import umath

log = logging.getLogger(__name__)

__all__ = ['estimate_radius','estimate_mass','estimate_logg']

"""Global parameters for the sun"""
numaxsol = ufloat(3090, 30) # microhertz | Huber et al. 2011
dnusol = ufloat(135.1, 0.1) # microhertz | Huber et al. 2011
teffsol = ufloat(5772., 0.8) # Kelvin    | Prša et al. 2016
gsol = ((const.G * const.M_sun)/(const.R_sun)**2).to(u.cm/u.second**2) #cms^2

def estimate_radius(numax, dnu, Teff, numax_err=None, dnu_err=None, Teff_err=None):
    """Calculates radius using the asteroseismic scaling relations. This method
    borrows heavily from work done in Bellinger et al. (2019).

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.

    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. If not given an astropy unit, assumed to be in units of
        microhertz.

    Teff : float
        The effective temperature of the star. In units of Kelvin.

    numax_err : float
        Error on numax. Assumed to be same units as numax

    dnu_err : float
        Error on dnu. Assumed to be same units as dnu

    Teff_err : float
        Error on Teff. Assumed to be same units as Teff.

    Returns
    -------
    R : float
        An estimate of the stellar radius in solar radii.

    R_e : float
        Uncertainty on the stellar radius estimate in solar radii. Returned only
        if all of `numax_err`, `dnu_err` and `teff_err` are passed.
    """
    numax = u.Quantity(numax, u.microhertz).value
    dnu = u.Quantity(dnu, u.microhertz).value
    Teff = u.Quantity(Teff, u. Kelvin).value

    if all(b is not None for b in [numax_err, dnu_err, Teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        dnu_err = u.Quantity(dnu_err, u.microhertz).value
        Teff_err = u.Quantity(Teff_err, u.Kelvin).value
        unumax = ufloat(numax, numax_err)
        udnu = ufloat(dnu, dnu_err)
        uTeff = ufloat(Teff, Teff_err)

    else:
        unumax = ufloat(numax, 0)
        udnu = ufloat(dnu, 0)
        uTeff = ufloat(Teff, 0)

    uR = (unumax / numaxsol) * (udnu / dnusol)**(-2.) * (uTeff / teffsol)**(0.5)

    R = uR.n * u.solRad
    R_e = uR.s * u.solRad

    if all(b is not None for b in [numax_err, dnu_err, Teff_err]):
        return R, R_e
    else:
        return R

def estimate_mass(numax, dnu, Teff, numax_err=None, dnu_err=None, Teff_err=None):
    """Calculates mass using the asteroseismic scaling relations. This method
    borrows heavily from work done in Bellinger et al. (2019).

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.

    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. If not given an astropy unit, assumed to be in units of
        microhertz.

    Teff : float
        The effective temperature of the star. In units of Kelvin.

    numax_err : float
        Error on numax. Assumed to be same units as numax

    dnu_err : float
        Error on dnu. Assumed to be same units as dnu

    Teff_err : float
        Error on Teff. Assumed to be same units as Teff.

    Returns
    -------
    M : float
        An estimate of the stellar mass in solar masses.

    M_e : float
        Uncertainty on the stellar mass estimate in solar masses. Returned only
        if all of `numax_err`, `dnu_err` and `teff_err` are passed.
    """
    numax = u.Quantity(numax, u.microhertz).value
    dnu = u.Quantity(dnu, u.microhertz).value
    Teff = u.Quantity(Teff, u.Kelvin).value

    if all(b is not None for b in [numax_err, dnu_err, Teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        dnu_err = u.Quantity(dnu_err, u.microhertz).value
        Teff_err = u.Quantity(Teff_err, u.Kelvin).value

        unumax = ufloat(numax, numax_err)
        udnu = ufloat(dnu, dnu_err)
        uTeff = ufloat(Teff, Teff_err)

    else:
        unumax = ufloat(numax, 0)
        udnu = ufloat(dnu, 0)
        uTeff = ufloat(Teff, 0)

    uM = (unumax / numaxsol)**3. * (udnu / dnusol)**(-4.) * (uTeff / teffsol)**(1.5)

    M = uM.n * u.solMass
    M_e = uM.s * u.solMass

    if all(b is not None for b in [numax_err, dnu_err, Teff_err]):
        return M, M_e
    else:
        return M

def estimate_logg(numax, Teff, numax_err=None, Teff_err=None):
    """Calculates the log of the surface gravity using the asteroseismic scaling
    relations. This method borrows heavily from work done in
    Bellinger et al. (2019).

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.

    Teff : float
        The effective temperature of the star. In units of Kelvin.

    numax_err : float
        Error on numax. Assumed to be same units as numax

    Teff_err : float
        Error on Teff. Assumed to be same units as Teff.

    Returns
    -------
    logg : float
        The log10 of the surface gravity of the star.

    logg_e : float
        Uncertainty on the log10 of the surface gravity in dex. Returned only
        if both of `numax_err` and `teff_err` are passed.
    """
    numax = u.Quantity(numax, u.microhertz).value
    Teff = u.Quantity(Teff, u.Kelvin).value

    if all(b is not None for b in [numax_err, Teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        Teff_err = u.Quantity(Teff_err, u.Kelvin).value

        unumax = ufloat(numax, numax_err)
        uTeff = ufloat(Teff, Teff_err)

    else:
        unumax = ufloat(numax, 0)
        uTeff = ufloat(Teff, 0)

    ug = gsol.value * (unumax / numaxsol) * (uTeff / teffsol)**0.5
    ulogg = umath.log(ug, 10)

    logg = ulogg.n * u.dex
    logg_e = ulogg.s * u.dex

    if all(b is not None for b in [numax_err, Teff_err]):
        return  logg, logg_e
    return logg
