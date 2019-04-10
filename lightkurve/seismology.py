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

log = logging.getLogger(__name__)

__all__ = ['dnu_mass_prior', 'estimate_radius', '_get_radius_err', 'estimate_mass',
            '_get_mass_err', 'estimate_logg','_get_logg_err']

"""Asteroseismic parameters"""
Numaxsol = u.Quantity(3090, u.microhertz) #[Huber et al 2011]
eNumaxsol = u.Quantity(30, u.microhertz) #[Huber et al 2011]
Dnusol = u.Quantity(135.1 , u.microhertz) #[Huber et al 2011]
eDnusol = u.Quantity(0.1, u.microhertz) #[Huber et al 2011]
"""Solar parameters"""
Tsol = 5777 #[Williams 2013]
Rsol = const.R_sun.to(u.R_sun)
Msol = const.M_sun.to(u.M_sun)
gsol = 100 * (const.G * const.M_sun)/(const.R_sun)**2 #cms^2

def dnu_mass_prior(numax, numax_sol=3050.0,
               dnu_sol=135.1, teff_sol=5777.0):
    """ Use the mass scaling relation to calculate a resonable range that could
    contain dnu. A ' indicates a solar value.

    M/M' = (numax/numax')**3 (dnu/dnu')**-4 (teff/teff')**(3/2), and so

    (dnu/dnu')**4 = numaxr**3 teffr**3/2 * Mr**-1, and so

    dnu = (nuamxr**3 * teffr**3/2 / Mr)**1/4 * dnu'
    """
    numaxr = numax / numax_sol
    teffr = np.array([3500, 5500]) / teff_sol    #A reasonable range of temperatures
    Mr = np.array([10.0, 0.1])                   #A reasonable range of masses
    dnu = (numaxr**3 * teffr**1.5 / Mr)**0.25 * dnu_sol   #Calculate range of dnus
    return dnu

#We can worry about unit conversions later
def estimate_radius(numax, dnu, Teff,
                    numax_err=None, dnu_err=None, Teff_err=None,
                    fdnu=1., fnumax=1.):
    """Calculates radius using the asteroseismic scaling relations.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. In units of microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    dnu_err : float
        Error on the dnu value.
    Teff_err : float
        Error on the Teff value,
    fdnu : int
        A correction to the seismic scaling relation for Delta Nu. Effectively
        rescales the solar value for Delta Nu.
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    R : float
        An estimate of the stellar radius.

    If any of `numax_err`, `dnu_err` and `teff_err` are passed, it will
    also return:
    sigR : float
        Uncertainty on the Radius estimate.
    """
    R = Rsol * (numax / (fnumax*Numaxsol)) * (dnu / (fdnu * Dnusol))**(-2.) * (Teff / Tsol)**(0.5)

    if not all(b is None for b in [numax_err, dnu_err, Teff_err]):
        return R, _get_radius_err(numax, numax_err, dnu, dnu_err,
                                    Teff, Teff_err, fdnu, fnumax)
    return R

def _get_radius_err(numax, numax_err, dnu, dnu_err,
                    Teff, Teff_err, fdnu, fnumax):
    """Return the uncertainty on the radius estimate, calcualted by propagating
    through uncertainties on properties that go into the seismic scaling
    relation for radius.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. In units of microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    dnu_err : float
        Error on the dnu value.
    Teff_err : float
        Error on the Teff value,
    fdnu : int
        A correction to the seismic scaling relation for Delta Nu. Effectively
        rescales the solar value for Delta Nu.
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    sigR : float
        Uncertainty on the Radius estimate.
    """

    #Calculate term for numax error
    if numax_err is not None:
        term = (Rsol/(fnumax * Numaxsol))*(dnu/(fdnu * Dnusol))**(-2)*(Teff/Tsol)**(0.5)
        drdnumax = term**2. * numax_err**2.
    else:
        drdnumax = 0.

    #Calculate term for dnu error
    if dnu_err is not None:
        term = (Rsol/((fdnu * Dnusol)**(-2.)))*(numax/(fnumax * Numaxsol))*(Teff/Tsol)**(0.5) * (-2.*dnu**(-3.))
        drdnu = term**2. * dnu_err**2.
    else:
        drdnu = 0.

    #Calculate term for temperature error
    if Teff_err is not None:
        term = (Rsol/Tsol**(0.5))*(numax/(fnumax * Numaxsol))*(dnu / (fdnu * Dnusol))**(-2.) * 0.5*Teff**(-0.5)
        drdt = term**2. * Teff_err**2.
    else:
        drdt = 0.

    #Calculate term for solar numax error
    term_nms = Rsol * (dnu/ (fdnu * Dnusol))**(-2.) * (Teff/Tsol)**(0.5) * (-1.*numax / ((fnumax * Numaxsol)**2.))
    drdnumaxsol = term_nms**2. * eNumaxsol**2.

    #Calculate term for solar dnu error
    term_dns = Rsol * (numax/(fnumax * Numaxsol)) * (Teff/Tsol)**(0.5) * (2.*fdnu*Dnusol)/(dnu**2.)
    drddnusol = term_dns**2. * eDnusol**2.

    sigR = np.sqrt(drdnumax + drdnu + drdt + drdnumaxsol + drddnusol)
    return sigR

def estimate_mass(numax, dnu, Teff,
                    numax_err=None, dnu_err=None, Teff_err=None,
                    fdnu=1., fnumax=1.):
    """Calculates mass using the asteroseismic scaling relations.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. In units of microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    dnu_err : float
        Error on the dnu value.
    Teff_err : float
        Error on the Teff value,
    fdnu : int
        A correction to the seismic scaling relation for Delta Nu. Effectively
        rescales the solar value for Delta Nu.
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    M : float
        An estimate of the stellar radius.

    If any of `numax_err`, `dnu_err` and `teff_err` are passed, it will
    also return:
    sigM : float
        Uncertainty on the Mass estimate.
    """
    M = Msol * (numax / (fnumax*Numaxsol))**3. * (dnu / (fdnu * Dnusol))**(-4.) * (Teff / Tsol)**(1.5)

    if not all(b is None for b in [numax_err, dnu_err, Teff_err]):
        return M, _get_mass_err(numax, numax_err, dnu, dnu_err,
                                    Teff, Teff_err, fdnu, fnumax)
    return M

def _get_mass_err(numax, numax_err, dnu, dnu_err,
                    Teff, Teff_err, fdnu, fnumax):
    """Return the uncertainty on the mass estimate, calcualted by propagating
    through uncertainties on properties that go into the seismic scaling
    relation for radius.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    dnu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. In units of microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    dnu_err : float
        Error on the dnu value.
    Teff_err : float
        Error on the Teff value,
    fdnu : int
        A correction to the seismic scaling relation for Delta Nu. Effectively
        rescales the solar value for Delta Nu.
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    sigM : float
        Uncertainty on the Mass estimate.
    """
    #Calculate term for numax error
    if numax_err is not None:
        term = (Msol/(fnumax * Numaxsol)**3.)*(dnu/(fdnu * Dnusol))**(-4.)*(Teff/Tsol)**(1.5) * 3.*numax**2.
        dmdnumax = term**2. * numax_err**2.
    else:
        dmdnumax = 0.

    #Calculate term for dnu error
    if dnu_err is not None:
        term = (Msol/((fdnu * Dnusol)**(-4.)))*(numax/(fnumax * Numaxsol))**3.*(Teff/Tsol)**(1.5) * (-4.*dnu**(-5.))
        dmdnu = term**2. * dnu_err**2.
    else:
        dmdnu = 0.

    #Calculate term for temperature error
    if Teff_err is not None:
        term = (Msol/Tsol**(1.5))*(numax/(fnumax * Numaxsol))**3.*(dnu /(fdnu * Dnusol))**(-4.) * 1.5*Teff**(0.5)
        dmdt = term**2. * Teff_err**2.
    else:
        dmdt = 0.

    #Calculate term for solar numax error
    term_nms = Msol * (dnu/(fdnu * Dnusol))**(-4.)*(Teff/Tsol)**(1.5) * (-3.*numax**3.)/((fnumax * Numaxsol)**4.)
    dmdnumaxsol = term_nms**2. * eNumaxsol**2.

    #Calculate term for solar dnu error
    term_dns = Msol * (numax/(fnumax * Numaxsol))**3.*(Teff/Tsol)**(1.5) * (4.*fdnu*Dnusol**3.)/(dnu**4.)
    dmddnusol = term_dns**2. * eDnusol**2.

    sigM = np.sqrt(dmdnumax + dmdnu + dmdt + dmdnumaxsol + dmddnusol)
    return sigM

def estimate_logg(numax, Teff, numax_err=None, Teff_err=None,
                            fnumax=1.):
    """Calculates the log of the surface gravity using the asteroseismic scaling
    relations.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    Teff_err : float
        Error on the Teff value,
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    logg : float
        The log10 of the surface gravity of the star.
    """

    g = gsol.value * (numax / (fnumax * Numaxsol)) * (Teff/Tsol)**0.5

    if not all(b is None for b in [numax_err, Teff_err]):
        return np.log10(g.value) * u.dex, _get_logg_err(numax, numax_err, Teff, Teff_err, fnumax)
    return np.log10(g.value) * u.dex

def _get_logg_err(numax, numax_err, Teff, Teff_err, fnumax):
    """Calculates the unceratinty on log of the surface gravity using the
    asteroseismic scaling relations.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. In units of
        microhertz.
    Teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on the numax value.
    Teff_err : float
        Error on the Teff value,
    fnumax : int
        A correction to the seismic scaling relation for Numax. Effectively
        rescales the solar value for Numax.

    Returns
    -------
    R : float
        An estimate of the stellar radius.

    If any of `numax_err` and `teff_err` are passed, it will
    also return:
    siglogg : float
        Uncertainty on the log(g) estimate.
    """
    #First we calculate the error on g
    if numax_err is not None:
        dgdnumax = ((gsol.value/(fnumax * Numaxsol))*(Teff/Tsol)**0.5)**2. * numax_err**2.
    else:
        dgdnumax = 0.

    if Teff_err is not None:
        dgdteff = ((gsol.value/Tsol**(0.5)) * (numax/(fnumax * Numaxsol)) * 0.5*Teff**(-0.5))**2. * Teff_err**2.
    else:
        dgdteff = 0.

    dgdnumaxsol = (gsol.value * numax * (Teff/Tsol)**0.5 * (-1./(fnumax * Numaxsol)**2.))**2. * eNumaxsol**2.

    sigg = np.sqrt(dgdnumax + dgdteff + dgdnumaxsol)

    #Then we convert to log10 space
    g = gsol * (numax/(fnumax * Numaxsol)) * (Teff/Tsol)**0.5
    siglogg = sigg.value / (g.value * np.log(10.)) * u.dex

    return siglogg
