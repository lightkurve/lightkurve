from __future__ import division, print_function

from astropy.visualization import (PercentileInterval, ImageNormalize,
                                   SqrtStretch, LogStretch, LinearStretch)
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np


__all__ = ['KeplerQualityFlags', 'TessQualityFlags', 'channel_to_module_output',
           'module_output_to_channel', 'running_mean']


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
    lookup[:,0] = 0
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


def bkjd_to_time(bkjd, timecorr, timslice, bjdref=2454833.):
    """Converts Barycentric Kepler Julian Day (BKJD) to an astropy.time.Time object.

    Kepler Barycentric Julian Day is a Julian day minus 2454833.0 (UTC=January
    1, 2009 12:00:00) and corrected to be the arrival times at the barycenter
    of the Solar System. See Section 2.3.2 in the Kepler Archive Manual.

    Parameters
    ----------
    bkjd : array of floats
        Barycentric Kepler Julian Day
    timecorr : array of floats
        Kepler barycentric correction
    timslice : array of floats
        Kepler time-slice correction
    bjdref : float
        BJD reference date, for Kepler this is 2454833

    Returns
    -------
    time : astropy.time.Time object
        Resulting time object
    """
    bjd = bkjd + bjdref
    jd = bjd - timecorr
    jd += (0.25 + 0.62 * (5 - timslice)) / 86400.
    return Time(jd, format='jd')


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
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            raise ValueError("scale {} is not available.".format(scale))

    cax = ax.imshow(image, origin=origin, norm=norm, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_colorbar:
        cbar = plt.colorbar(cax, ax=ax, norm=norm, label=clabel)
    return ax
