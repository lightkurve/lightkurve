from __future__ import division, print_function
import sys

from astropy.visualization import (PercentileInterval, ImageNormalize,
                                   SqrtStretch, LinearStretch)
from astropy.time import Time
from astroquery.vizier import Vizier
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import logging

log = logging.getLogger(__name__)


__all__ = ['KeplerQualityFlags', 'TessQualityFlags',
           'bkjd_to_astropy_time', 'btjd_to_astropy_time',
           'channel_to_module_output', 'module_output_to_channel',
           'running_mean']


class KeplerQualityFlags(object):
    """
    This class encodes the meaning of the various Kepler QUALITY bitmask flags,
    as documented in the Kepler Archive Manual (Table 2.3).
    """
    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    ZeroCrossing = 16
    Desat = 32
    Argabrightening = 64
    ApertureCosmic = 128
    ManualExclude = 256
    SensitivityDropout = 1024
    ImpulsiveOutlier = 2048
    ArgabrighteningOnCCD = 4096
    CollateralCosmic = 8192
    DetectorAnomaly = 16384
    NoFinePoint = 32768
    NoData = 65536
    RollingBandInAperture = 131072
    RollingBandInMask = 262144
    PossibleThrusterFiring = 524288
    ThrusterFiring = 1048576

    # Which is the recommended QUALITY mask to identify bad data?
    DEFAULT_BITMASK = (AttitudeTweak | SafeMode | CoarsePoint | EarthPoint |
                       Desat | ManualExclude |
                       DetectorAnomaly | NoData | ThrusterFiring)

    # This bitmask includes flags that are known to identify both good and bad cadences.
    # Use it wisely.
    HARD_BITMASK = (DEFAULT_BITMASK | SensitivityDropout | ApertureCosmic |
                    CollateralCosmic | PossibleThrusterFiring)

    # Using this bitmask only QUALITY == 0 cadences will remain
    HARDEST_BITMASK = 2096639

    # Give the recommended bitmask options friendly names
    OPTIONS = {'default': DEFAULT_BITMASK,
               'hard': HARD_BITMASK,
               'hardest': HARDEST_BITMASK}

    # Pretty string descriptions for each flag
    STRINGS = {
        1: "Attitude tweak",
        2: "Safe mode",
        4: "Coarse point",
        8: "Earth point",
        16: "Zero crossing",
        32: "Desaturation event",
        64: "Argabrightening",
        128: "Cosmic ray in optimal aperture",
        256: "Manual exclude",
        1024: "Sudden sensitivity dropout",
        2048: "Impulsive outlier",
        4096: "Argabrightening on CCD",
        8192: "Cosmic ray in collateral data",
        16384: "Detector anomaly",
        32768: "No fine point",
        65536: "No data",
        131072: "Rolling band in optimal aperture",
        262144: "Rolling band in full mask",
        524288: "Possible thruster firing",
        1048576: "Thruster firing"
    }

    @classmethod
    def decode(cls, quality):
        """Converts a Kepler QUALITY value into a list of human-readable strings.

        This function takes the QUALITY bitstring that can be found for each
        cadence in Kepler/K2's pixel and light curve files and converts into
        a list of human-readable strings explaining the flags raised (if any).

        Parameters
        ----------
        quality : int
            Value from the 'QUALITY' column of a Kepler/K2 pixel or lightcurve file.

        Returns
        -------
        flags : list of str
            List of human-readable strings giving a short description of the
            quality flags raised.  Returns an empty list if no flags raised.
        """
        result = []
        for flag in cls.STRINGS.keys():
            if quality & flag > 0:
                result.append(cls.STRINGS[flag])
        return result


class TessQualityFlags(KeplerQualityFlags):
    pass


def channel_to_module_output(channel):
    """Returns a (module, output) pair given a CCD channel number.

    Parameters
    ----------
    channel : int
        Channel number

    Returns
    -------
    module, output : tuple of ints
        Module and Output number
    """
    if channel < 1 or channel > 88:
        raise ValueError("Channel number must be in the range 1-88.")
    lookup = _get_channel_lookup_array()
    lookup[:, 0] = 0
    modout = np.where(lookup == channel)
    return (modout[0][0], modout[1][0])


def module_output_to_channel(module, output):
    """Returns the CCD channel number for a given module and output pair.

    Parameters
    ----------
    module : int
        Module number
    output : int
        Output number

    Returns
    -------
    channel : int
        Channel number
    """
    if module < 1 or module > 26:
        raise ValueError("Module number must be in range 1-26.")
    if output < 1 or output > 4:
        raise ValueError("Output number must be 1, 2, 3, or 4.")
    return _get_channel_lookup_array()[module, output]


def _get_channel_lookup_array():
    """Returns a lookup table which maps (module, output) onto channel."""
    # In the array below, channel == array[module][output]
    # Note: modules 1, 5, 21, 25 are the FGS guide star CCDs.
    return np.array([
        [0,     0,    0,    0,    0],
        [1,    85,    0,    0,    0],
        [2,     1,    2,    3,    4],
        [3,     5,    6,    7,    8],
        [4,     9,   10,   11,   12],
        [5,    86,    0,    0,    0],
        [6,    13,   14,   15,   16],
        [7,    17,   18,   19,   20],
        [8,    21,   22,   23,   24],
        [9,    25,   26,   27,   28],
        [10,   29,   30,   31,   32],
        [11,   33,   34,   35,   36],
        [12,   37,   38,   39,   40],
        [13,   41,   42,   43,   44],
        [14,   45,   46,   47,   48],
        [15,   49,   50,   51,   52],
        [16,   53,   54,   55,   56],
        [17,   57,   58,   59,   60],
        [18,   61,   62,   63,   64],
        [19,   65,   66,   67,   68],
        [20,   69,   70,   71,   72],
        [21,   87,    0,    0,    0],
        [22,   73,   74,   75,   76],
        [23,   77,   78,   79,   80],
        [24,   81,   82,   83,   84],
        [25,   88,    0,    0,    0],
    ])


def running_mean(data, window_size):
    """Returns the moving average of an array `data`.

    Parameters
    ----------
    data : array of numbers
        The running mean will be computed on this data.
    window_size : int
        Window length used to compute the running mean.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def bkjd_to_astropy_time(bkjd, bjdref=2454833.):
    """Converts BKJD time values to an `astropy.time.Time` object.

    Kepler Barycentric Julian Day (BKJD) is a Julian day minus 2454833.0
    (UTC=January 1, 2009 12:00:00) and corrected to the arrival times
    at the barycenter of the Solar System.
    BKJD is the format in which times are recorded in the Kepler data products.
    The time is in the Barycentric Dynamical Time frame (TDB), which is a
    time system that is not affected by leap seconds.
    See Section 2.3.2 in the Kepler Archive Manual for details.

    Parameters
    ----------
    bkjd : array of floats
        Barycentric Kepler Julian Day.
    bjdref : float
        BJD reference date, for Kepler this is 2454833.

    Returns
    -------
    time : astropy.time.Time object
        Resulting time object
    """
    jd = bkjd + bjdref
    # Some data products have missing time values;
    # we need to set these to zero or `Time` cannot be instantiated.
    jd[~np.isfinite(jd)] = 0
    return Time(jd, format='jd', scale='tdb')


def btjd_to_astropy_time(btjd, bjdref=2457000.):
    """Converts BTJD time values to an `astropy.time.Time` object.

    TESS Barycentric Julian Day (BTJD) is a Julian day minus 2457000.0
    and corrected to the arrival times at the barycenter of the Solar System.
    BTJD is the format in which times are recorded in the TESS data products.
    The time is in the Barycentric Dynamical Time frame (TDB), which is a
    time system that is not affected by leap seconds.

    Parameters
    ----------
    btjd : array of floats
        Barycentric Kepler Julian Day
    bjdref : float
        BJD reference date.

    Returns
    -------
    time : astropy.time.Time object
        Resulting time object
    """
    jd = btjd + bjdref
    jd[~np.isfinite(jd)] = 0
    return Time(jd, format='jd', scale='tdb')


def plot_image(image, ax=None, scale='linear', origin='lower',
               xlabel='Pixel Column Number', ylabel='Pixel Row Number',
               clabel='Flux ($e^{-}s^{-1}$)', title=None, show_colorbar=True,
               **kwargs):
    """Utility function to plot a 2D image

    Parameters
    ----------
    image : 2d array
        Image data.
    ax : matplotlib.axes._subplots.AxesSubplot
        A matplotlib axes object to plot into. If no axes is provided,
        a new one will be generated.
    scale : str
        Scale used to stretch the colormap.
        Options: 'linear', 'sqrt', or 'log'.
    origin : str
        The origin of the coordinate system.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    clabel : str
        Label for the color bar.
    title : str or None
        Title for the plot.
    show_colorbar : bool
        Whether or not to show the colorbar
    kwargs : dict
        Keyword arguments to be passed to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    if ax is None:
        _, ax = plt.subplots()
    vmin, vmax = PercentileInterval(95.).get_limits(image)

    norm = None
    if scale is not None:
        if scale == 'linear':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif scale == 'sqrt':
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        elif scale == 'log':
            # To use log scale we need to guarantee that vmin > 0, so that
            # we avoid division by zero and/or negative values.
            norm = LogNorm(vmin=max(vmin, sys.float_info.epsilon), vmax=vmax,
                           clip=True)
        else:
            raise ValueError("scale {} is not available.".format(scale))

    cax = ax.imshow(image, origin=origin, norm=norm, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_colorbar:
        plt.colorbar(cax, ax=ax, norm=norm, label=clabel)
    return ax


def query_catalog(coordinate, catalog="KIC", radius=30.):
        """Returns an astropy table of sources obtained using a conesearch query.

        This helper function allows the key catalogs that are relevant to
        Kepler/K2/TESS to be queried such that they return uniform columns.
        Current catalogs supported are 'KIC', 'EPIC', and 'Gaia2'.

        Parameters
        -----------
        coordinate : astropy.coordinates.SkyCoord object
            Coordinate to query around.
        catalog: 'KIC', 'EPIC', or 'Gaia2'
            Which catalog to query?
        radius: float or astropy.units.Quantity object
            Radius of cone search centered on the target. If a float is given
            then it is assumed to be in units arcseconds. (Default: 30 arcsec.)

        Returns
        -------
        result : astropy.table.Table object
            Astropy table with the following columns:
            - id : Catalog ID from catalog.
            - ra: Right ascension [degrees]
            - dec: Declination [degrees]
            - pmra: Proper motion for right ascension [mas/year]
            - pmdec: Proper motion for declination [mas/year]
            - mag: Magnitude in the Kepler or Gaia band [mag]
        """

        # First, validate and convert the inputs:
        # If the search radius is a float, assume it is in unit arcseconds
        if not isinstance(radius, u.Quantity):
            radius = u.Quantity(radius, u.arcsec)
        # Use lowercase catalog names to avoid case sensitivity
        catalog = catalog.lower()
        # Ensure `catalog` is one of our supported catalogs
        supported_catalogs = \
            {"kic":
             {"vizier": "V/133/kic",
              "cols": ["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "kepmag"],
              "epoch": 2000.},
             "epic":
             {"vizier": "IV/34/epic",
              "cols": ["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "Kpmag"],
              "epoch": 2000.},
             "gaia2":
             {"vizier": "I/345/gaia2",
              "cols": ["DR2Name", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "Gmag"],
              "epoch": 2015.5}}
        if catalog not in supported_catalogs.keys():
            raise ValueError('catalog must be one of {}'.format(list(supported_catalogs.keys())))

         # Query Vizier
        vizier_id = supported_catalogs[catalog]['vizier']
        v = Vizier(catalog=[vizier_id], columns=supported_catalogs[catalog]['cols'])
        result = v.query_region(coordinate, radius=radius, catalog=vizier_id)
        if len(result) == 0:
            log.info('No sources found in queried region. Try another catalog.')

        # Rename columns to be uniform for all catalogs
        new_pars = ['id', 'ra', 'dec', 'pmra', 'pmdec', 'mag']
        for i in range(len(new_pars)):
            result[vizier_id].rename_column(result[vizier_id].colnames[i], new_pars[i])

         return result[vizier_id]

def kpmag_to_flux(kpmag):
    """
    Returns the predicted flux in e-/s of a given Kp magnitude source,
    estimated by pre-flight zero-point.
    Source: Kepler Science Center.
    """
    f12 = 1.74 * 10**5  # pre-flight zero point
    exponent = -0.4*(kpmag - 12)
    return 10**(exponent) * f12
