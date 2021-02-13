"""Functions to estimate stellar parameters (radius, mass, logg) using
asteroseismic scaling relations.
"""
from uncertainties import ufloat, umath

from astropy import units as u
from astropy import constants as const

from .utils import SeismologyQuantity

__all__ = ["estimate_radius", "estimate_mass", "estimate_logg"]


"""Global parameters for the sun"""
NUMAX_SOL = ufloat(3090, 30)  # microhertz | Huber et al. 2011
DELTANU_SOL = ufloat(135.1, 0.1)  # microhertz | Huber et al. 2011
TEFF_SOL = ufloat(5772.0, 0.8)  # Kelvin    | Prsa et al. 2016
G_SOL = ((const.G * const.M_sun) / (const.R_sun) ** 2).to(u.cm / u.second ** 2)  # cms^2


def estimate_radius(
    numax, deltanu, teff, numax_err=None, deltanu_err=None, teff_err=None
):
    """Returns a stellar radius estimate based on the scaling relations.

    The two global observable seismic parameters, numax and deltanu, along with
    temperature, scale with fundamental stellar properties (Brown et al. 1991;
    Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
    calculate a stellar radius as

    R = Rsol * (numax/numax_sol)(deltanu/deltanusol)^-2(Teff/Teffsol)^0.5

    where R is the radius and Teff is the effective temperature, and the suffix
    'sol' indicates a solar value. In this method we use the solar values for
    numax and deltanu as given in Huber et al. (2011) and for Teff as given in
    Prsa et al. (2016).

    This code structure borrows from work done in Bellinger et al. (2019), which
    also functions as an accessible explanation of seismic scaling relations.

    If no value of effective temperature is given, this function will check the
    meta data of the `Periodogram` object used to create the `Seismology` object.
    These data will often contain an effective tempearture from the Kepler Input
    Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
    or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
    catalogues are estimated using photometry, and so have large associated uncertainties
    (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
    temperature are often more precise.

    NOTE: These scaling relations are scaled to the Sun, and therefore do not
    always produce an entirely accurate result for more evolved stars.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.
    deltanu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. If not given an astropy unit, assumed to be in units of
        microhertz.
    teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on numax. Assumed to be same units as numax
    deltanu_err : float
        Error on deltanu. Assumed to be same units as deltanu
    teff_err : float
        Error on Teff. Assumed to be same units as Teff.

    Returns
    -------
    radius : SeismologyQuantity
        An estimate of the stellar radius in solar radii.
    """
    numax = u.Quantity(numax, u.microhertz).value
    deltanu = u.Quantity(deltanu, u.microhertz).value
    teff = u.Quantity(teff, u.Kelvin).value

    if all(b is not None for b in [numax_err, deltanu_err, teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        deltanu_err = u.Quantity(deltanu_err, u.microhertz).value
        teff_err = u.Quantity(teff_err, u.Kelvin).value
        unumax = ufloat(numax, numax_err)
        udeltanu = ufloat(deltanu, deltanu_err)
        uteff = ufloat(teff, teff_err)
    else:
        unumax = ufloat(numax, 0)
        udeltanu = ufloat(deltanu, 0)
        uteff = ufloat(teff, 0)

    uR = (
        (unumax / NUMAX_SOL)
        * (udeltanu / DELTANU_SOL) ** (-2.0)
        * (uteff / TEFF_SOL) ** (0.5)
    )
    result = SeismologyQuantity(
        uR.n * u.solRad,
        error=uR.s * u.solRad,
        name="radius",
        method="Uncorrected Scaling Relations",
    )
    return result


def estimate_mass(
    numax, deltanu, teff, numax_err=None, deltanu_err=None, teff_err=None
):
    """Calculates mass using the asteroseismic scaling relations.

    The two global observable seismic parameters, numax and deltanu, along with
    temperature, scale with fundamental stellar properties (Brown et al. 1991;
    Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
    calculate a stellar mass as

    M = Msol * (numax/numax_sol)^3(deltanu/deltanusol)^-4(Teff/Teffsol)^1.5

    where M is the mass and Teff is the effective temperature, and the suffix
    'sol' indicates a solar value. In this method we use the solar values for
    numax and deltanu as given in Huber et al. (2011) and for Teff as given in
    Prsa et al. (2016).

    This code structure borrows from work done in Bellinger et al. (2019), which
    also functions as an accessible explanation of seismic scaling relations.

    If no value of effective temperature is given, this function will check the
    meta data of the `Periodogram` object used to create the `Seismology` object.
    These data will often contain an effective tempearture from the Kepler Input
    Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
    or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
    catalogues are estimated using photometry, and so have large associated uncertainties
    (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
    temperature are often more precise.

    NOTE: These scaling relations are scaled to the Sun, and therefore do not
    always produce an entirely accurate result for more evolved stars.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.
    deltanu : float
        The frequency spacing between two consecutive overtones of equal radial
        degree. If not given an astropy unit, assumed to be in units of
        microhertz.
    teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on numax. Assumed to be same units as numax
    deltanu_err : float
        Error on deltanu. Assumed to be same units as deltanu
    teff_err : float
        Error on Teff. Assumed to be same units as Teff.

    Returns
    -------
    mass : SeismologyQuantity
        An estimate of the stellar mass in solar masses.
    """
    numax = u.Quantity(numax, u.microhertz).value
    deltanu = u.Quantity(deltanu, u.microhertz).value
    teff = u.Quantity(teff, u.Kelvin).value

    if all(b is not None for b in [numax_err, deltanu_err, teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        deltanu_err = u.Quantity(deltanu_err, u.microhertz).value
        teff_err = u.Quantity(teff_err, u.Kelvin).value

        unumax = ufloat(numax, numax_err)
        udeltanu = ufloat(deltanu, deltanu_err)
        uteff = ufloat(teff, teff_err)
    else:
        unumax = ufloat(numax, 0)
        udeltanu = ufloat(deltanu, 0)
        uteff = ufloat(teff, 0)

    uM = (
        (unumax / NUMAX_SOL) ** 3.0
        * (udeltanu / DELTANU_SOL) ** (-4.0)
        * (uteff / TEFF_SOL) ** (1.5)
    )
    result = SeismologyQuantity(
        uM.n * u.solMass,
        error=uM.s * u.solMass,
        name="mass",
        method="Uncorrected Scaling Relations",
    )
    return result


def estimate_logg(numax, teff, numax_err=None, teff_err=None):
    """Calculates the log of the surface gravity using the asteroseismic scaling
    relations.

    The two global observable seismic parameters, numax and deltanu, along with
    temperature, scale with fundamental stellar properties (Brown et al. 1991;
    Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
    calculate a stellar surface gravity as

        g = gsol * (numax/numax_sol)(Teff/Teffsol)^0.5

    where g is the surface gravity and Teff is the effective temperature,
    and the suffix 'sol' indicates a solar value. In this method we use the
    solar values for numax as given in Huber et al. (2011) and for Teff as given
    in Prsa et al. (2016). The solar surface gravity is calcluated from the
    astropy constants for solar mass and radius and does not have an error.

    The solar surface gravity is returned as log10(g) with units in dex, as is
    common in the astrophysics literature.

    This code structure borrows from work done in Bellinger et al. (2019), which
    also functions as an accessible explanation of seismic scaling relations.

    If no value of effective temperature is given, this function will check the
    meta data of the `Periodogram` object used to create the `Seismology` object.
    These data will often contain an effective tempearture from the Kepler Input
    Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
    or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
    catalogues are estimated using photometry, and so have large associated uncertainties
    (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
    temperature are often more precise.

    NOTE: These scaling relations are scaled to the Sun, and therefore do not
    always produce an entirely accurate result for more evolved stars.

    Parameters
    ----------
    numax : float
        The frequency of maximum power of the seismic mode envelope. If not
        given an astropy unit, assumed to be in units of microhertz.
    teff : float
        The effective temperature of the star. In units of Kelvin.
    numax_err : float
        Error on numax. Assumed to be same units as numax
    teff_err : float
        Error on teff. Assumed to be same units as teff.

    Returns
    -------
    logg : `.SeismologyQuantity`
        The log10 of the surface gravity of the star.
    """
    numax = u.Quantity(numax, u.microhertz).value
    teff = u.Quantity(teff, u.Kelvin).value

    if all(b is not None for b in [numax_err, teff_err]):
        numax_err = u.Quantity(numax_err, u.microhertz).value
        teff_err = u.Quantity(teff_err, u.Kelvin).value
        unumax = ufloat(numax, numax_err)
        uteff = ufloat(teff, teff_err)
    else:
        unumax = ufloat(numax, 0)
        uteff = ufloat(teff, 0)

    ug = G_SOL.value * (unumax / NUMAX_SOL) * (uteff / TEFF_SOL) ** 0.5
    ulogg = umath.log(ug, 10)

    result = SeismologyQuantity(
        ulogg.n * u.dex,
        error=ulogg.s * u.dex,
        name="logg",
        method="Uncorrected Scaling Relations",
    )
    return result
