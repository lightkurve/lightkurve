"""This module provides various helper functions."""
import logging
import sys
import os
import warnings
from functools import wraps
import urllib

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm

import astropy
from astropy.utils.data import download_file
from astropy.units.quantity import Quantity
import astropy.units as u
from astropy.visualization import (
    PercentileInterval,
    ImageNormalize,
    SqrtStretch,
    LinearStretch,
)
from astropy.time import Time


log = logging.getLogger(__name__)


__all__ = [
    "LightkurveError",
    "LightkurveWarning",
    "KeplerQualityFlags",
    "TessQualityFlags",
    "bkjd_to_astropy_time",
    "btjd_to_astropy_time",
    "show_citation_instructions",
    "finalize_notebook_url",
    "remote_jupyter_proxy_url"
]


class QualityFlags(object):
    """Abstract class"""

    STRINGS = {}
    OPTIONS = {}

    @classmethod
    def decode(cls, quality):
        """Converts a QUALITY value into a list of human-readable strings.

        This function takes the QUALITY bitstring that can be found for each
        cadence in Kepler/K2/TESS' pixel and light curve files and converts into
        a list of human-readable strings explaining the flags raised (if any).

        Parameters
        ----------
        quality : int
            Value from the 'QUALITY' column of a Kepler/K2/TESS pixel or lightcurve file.

        Returns
        -------
        flags : list of str
            List of human-readable strings giving a short description of the
            quality flags raised.  Returns an empty list if no flags raised.
        """
        # If passed an astropy quantity object, get the value
        if isinstance(quality, Quantity):
            quality = quality.value
        result = []
        for flag in cls.STRINGS.keys():
            if quality & flag > 0:
                result.append(cls.STRINGS[flag])
        return result

    @classmethod
    def create_quality_mask(cls, quality_array, bitmask=None):
        """Returns a boolean array which flags good cadences given a bitmask.

        This method is used by the readers of :class:`KeplerTargetPixelFile`
        and :class:`KeplerLightCurve` to initialize their `quality_mask`
        class attribute which is used to ignore bad-quality data.

        Parameters
        ----------
        quality_array : array of int
            'QUALITY' column of a Kepler target pixel or lightcurve file.
        bitmask : int or str
            Bitmask (int) or one of 'none', 'default', 'hard', or 'hardest'.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        # Return an array filled with `True` by default (i.e. ignore nothing)
        if bitmask is None:
            return np.ones(len(quality_array), dtype=bool)
        if isinstance(quality_array, u.Quantity):
            quality_array = quality_array.value
        # A few pre-defined bitmasks can be specified as strings
        if isinstance(bitmask, str):
            try:
                bitmask = cls.OPTIONS[bitmask]
            except KeyError:
                valid_options = tuple(cls.OPTIONS.keys())
                raise ValueError(
                    "quality_bitmask='{}' is not supported, "
                    "expected one of {}"
                    "".format(bitmask, valid_options)
                )
        # The bitmask is applied using the bitwise AND operator
        quality_mask = (quality_array & bitmask) == 0
        # Log the quality masking as info or warning
        n_cadences = len(quality_array)
        n_cadences_masked = (~quality_mask).sum()
        percent_masked = 100.0 * n_cadences_masked / n_cadences
        logmsg = (
            "{:.0f}% ({}/{}) of the cadences will be ignored due to the "
            "quality mask (quality_bitmask={})."
            "".format(percent_masked, n_cadences_masked, n_cadences, bitmask)
        )
        if percent_masked > 20:
            log.warning("Warning: " + logmsg)
        else:
            log.info(logmsg)
        return quality_mask


class KeplerQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various Kepler QUALITY bitmask flags,
    as documented in the Kepler Archive Manual (Ref. [1], Table 2.3).

    References
    ----------
    .. [1] Kepler: A Search for Terrestrial Planets. Kepler Archive Manual.
        http://archive.stsci.edu/kepler/manuals/archive_manual.pdf
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
    # Bit 2**10 = 512 is unused by Kepler
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

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    DEFAULT_BITMASK = (
        AttitudeTweak
        | SafeMode
        | CoarsePoint
        | EarthPoint
        | Desat
        | ManualExclude
        | DetectorAnomaly
        | NoData
        | ThrusterFiring
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK
        | SensitivityDropout
        | ApertureCosmic
        | CollateralCosmic
        | PossibleThrusterFiring
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 2096639

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }

    #: Pretty string descriptions for each flag
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
        1048576: "Thruster firing",
    }


class TessQualityFlags(QualityFlags):
    """
    This class encodes the meaning of the various TESS QUALITY bitmask flags,
    as documented in the TESS Data Products Description Document (Ref. [1], Table 28).

    References
    ----------
    .. [1] TESS Science Data Products Description Document (EXP-TESS-ARC-ICD-0014)
        https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014.pdf
    """

    AttitudeTweak = 1
    SafeMode = 2
    CoarsePoint = 4
    EarthPoint = 8
    Argabrightening = 16
    Desat = 32
    ApertureCosmic = 64
    ManualExclude = 128
    Discontinuity = 256
    ImpulsiveOutlier = 512
    CollateralCosmic = 1024
    #: The first stray light flag is set manually by MIT based on visual inspection.
    Straylight = 2048
    #: The second stray light flag is set automatically by Ames/SPOC based on background level thresholds.
    Straylight2 = 4096
    # See TESS Science Data Products Description Document
    PlanetSearchExclude = 8192
    BadCalibrationExclude = 16384
    # Set in the sector 20 data release notes
    InsufficientTargets = 32768

    #: DEFAULT bitmask identifies all cadences which are definitely useless.
    DEFAULT_BITMASK = (
        AttitudeTweak | SafeMode | CoarsePoint | EarthPoint | Desat | ManualExclude
    )
    #: HARD bitmask is conservative and may identify cadences which are useful.
    HARD_BITMASK = (
        DEFAULT_BITMASK | ApertureCosmic | CollateralCosmic | Straylight | Straylight2
    )
    #: HARDEST bitmask identifies cadences with any flag set. Its use is not recommended.
    HARDEST_BITMASK = 65535

    #: Dictionary which provides friendly names for the various bitmasks.
    OPTIONS = {
        "none": 0,
        "default": DEFAULT_BITMASK,
        "hard": HARD_BITMASK,
        "hardest": HARDEST_BITMASK,
    }

    #: Pretty string descriptions for each flag
    STRINGS = {
        1: "Attitude tweak",
        2: "Safe mode",
        4: "Coarse point",
        8: "Earth point",
        16: "Argabrightening",
        32: "Desaturation event",
        64: "Cosmic ray in optimal aperture",
        128: "Manual exclude",
        256: "Discontinuity corrected",
        512: "Impulsive outlier",
        1024: "Cosmic ray in collateral data",
        2048: "Straylight",
        4096: "Straylight2",
        8192: "Planet Search Exclude",
        16384: "Bad Calibration Exclude",
        32768: "Insufficient Targets for Error Correction Exclude",
    }


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
    return modout[0][0], modout[1][0]


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
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 85, 0, 0, 0],
            [2, 1, 2, 3, 4],
            [3, 5, 6, 7, 8],
            [4, 9, 10, 11, 12],
            [5, 86, 0, 0, 0],
            [6, 13, 14, 15, 16],
            [7, 17, 18, 19, 20],
            [8, 21, 22, 23, 24],
            [9, 25, 26, 27, 28],
            [10, 29, 30, 31, 32],
            [11, 33, 34, 35, 36],
            [12, 37, 38, 39, 40],
            [13, 41, 42, 43, 44],
            [14, 45, 46, 47, 48],
            [15, 49, 50, 51, 52],
            [16, 53, 54, 55, 56],
            [17, 57, 58, 59, 60],
            [18, 61, 62, 63, 64],
            [19, 65, 66, 67, 68],
            [20, 69, 70, 71, 72],
            [21, 87, 0, 0, 0],
            [22, 73, 74, 75, 76],
            [23, 77, 78, 79, 80],
            [24, 81, 82, 83, 84],
            [25, 88, 0, 0, 0],
        ]
    )


def running_mean(data, window_size):
    """Returns the moving average of an array `data`.

    Parameters
    ----------
    data : array of numbers
        The running mean will be computed on this data.
    window_size : int
        Window length used to compute the running mean.
    """
    if window_size > len(data):
        window_size = len(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def bkjd_to_astropy_time(bkjd) -> Time:
    """Converts Kepler Barycentric Julian Day (BKJD) time values to an
    `astropy.time.Time` object.

    Kepler Barycentric Julian Day (BKJD) is a Julian day minus 2454833.0
    (UTC=January 1, 2009 12:00:00) and corrected to the arrival times
    at the barycenter of the Solar System.
    BKJD is the format in which times are recorded in the Kepler data products.
    The time is in the Barycentric Dynamical Time frame (TDB), which is a
    time system that is not affected by leap seconds.
    See Section 2.3.2 in the Kepler Archive Manual for details.

    Parameters
    ----------
    bkjd : float or array of floats
        Barycentric Kepler Julian Day.

    Returns
    -------
    time : `astropy.time.Time` object
        Resulting time object.
    """
    bkjd = np.atleast_1d(bkjd)
    # Some data products have missing time values;
    # we need to set these to zero or `Time` cannot be instantiated.
    bkjd[~np.isfinite(bkjd)] = 0
    return Time(bkjd, format="bkjd", scale="tdb")


def btjd_to_astropy_time(btjd) -> Time:
    """Converts TESS Barycentric Julian Day (BTJD) values to an
    `astropy.time.Time` object.

    TESS Barycentric Julian Day (BTJD) is a Julian day minus 2457000.0
    and corrected to the arrival times at the barycenter of the Solar System.
    BTJD is the format in which times are recorded in the TESS data products.
    The time is in the Barycentric Dynamical Time frame (TDB), which is a
    time system that is not affected by leap seconds.

    Parameters
    ----------
    btjd : float or array of floats
        Barycentric TESS Julian Day

    Returns
    -------
    time : `astropy.time.Time` object
        Resulting time object.
    """
    btjd = np.atleast_1d(btjd)
    btjd[~np.isfinite(btjd)] = 0
    return Time(btjd, format="btjd", scale="tdb")


def plot_image(
    image,
    ax=None,
    scale="linear",
    origin="lower",
    xlabel="Pixel Column Number",
    ylabel="Pixel Row Number",
    clabel="Flux ($e^{-}s^{-1}$)",
    title=None,
    show_colorbar=True,
    vmin=None,
    vmax=None,
    **kwargs
):
    """Utility function to plot a 2D image

    Parameters
    ----------
    image : 2d array
        Image data.
    ax : `~matplotlib.axes.Axes`
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
    vmin : float
        Minimum colorbar value. By default, the 2.5%-percentile is used.
    vmax : float
        Maximum colorbar value. By default, the 97.5%-percentile is used.
    kwargs : dict
        Keyword arguments to be passed to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The matplotlib axes object.
    """
    if isinstance(image, u.Quantity):
        image = image.value
    if ax is None:
        _, ax = plt.subplots()

    if vmin is None or vmax is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # ignore image NaN values
            mask = np.nan_to_num(image) > 0
            if mask.any() > 0:
                vmin_default, vmax_default = PercentileInterval(95.0).get_limits(
                    image[mask]
                )
            else:
                vmin_default, vmax_default = 0, 0
            if vmin is None:
                vmin = vmin_default
            if vmax is None:
                vmax = vmax_default

    norm = None
    if scale is not None:
        if scale == "linear":
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax, stretch=LinearStretch(), clip=False
            )
        elif scale == "sqrt":
            norm = ImageNormalize(
                vmin=vmin, vmax=vmax, stretch=SqrtStretch(), clip=False
            )
        elif scale == "log":
            # To use log scale we need to guarantee that vmin > 0, so that
            # we avoid division by zero and/or negative values.
            norm = LogNorm(vmin=max(vmin, sys.float_info.epsilon), vmax=vmax, clip=True)
        else:
            raise ValueError("scale {} is not available.".format(scale))
    cax = ax.imshow(image, origin=origin, norm=norm, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_colorbar:
        cbar = plt.colorbar(cax, ax=ax, label=clabel)
        cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
        cbar.ax.minorticks_off()
    return ax


class LightkurveError(Exception):
    """Class for Lightkurve exceptions."""

    pass


class LightkurveWarning(Warning):
    """Class for all Lightkurve warnings."""

    pass


class LightkurveDeprecationWarning(LightkurveWarning):
    """Class for all Lightkurve deprecation warnings."""

    pass


def suppress_stdout(f, *args, **kwargs):
    """A simple decorator to suppress function print outputs."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        # redirect output to `null`
        with open(os.devnull, "w") as devnull:
            old_out = sys.stdout
            sys.stdout = devnull
            try:
                return f(*args, **kwargs)
            # restore to default
            finally:
                sys.stdout = old_out

    return wrapper


def validate_method(method, supported_methods):
    """Raises a `ValueError` if a method is not supported.

    Parameters
    ----------
    method : str
        The method specified by the user.
    supported_methods : list of str
        The methods supported.  All method names must be lowercase.

    Returns
    -------
    method : str
        Will return the method name if it is supported.
    """
    method = method.lower()
    if method in supported_methods:
        return method
    raise ValueError(
        "method '{}' is not supported; "
        "must be one of {}".format(method, supported_methods)
    )


def centroid_quadratic(data, mask=None):
    """Computes the quadratic estimate of the centroid in a 2d-array.

    This method will fit a simple 2D second-order polynomial
    $P(x, y) = a + bx + cy + dx^2 + exy + fy^2$
    to the 3x3 patch of pixels centered on the brightest pixel within
    the image.  This function approximates the core of the Point
    Spread Function (PSF) using a bivariate quadratic function, and returns
    the maximum (x, y) coordinate of the function using linear algebra.

    For the motivation and the details around this technique, please refer
    to Vakili, M., & Hogg, D. W. 2016, ArXiv, 1610.05873.

    Caveat: if the brightest pixel falls on the edge of the data array, the fit
    will tend to fail or be inaccurate.

    Parameters
    ----------
    data : 2D array
        The 2D input array representing the pixel values of the image.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as `data`, where a **True** value
        indicates the corresponding element of data is masked.

    Returns
    -------
    column, row : tuple
        The coordinates of the centroid in column and row.  If the fit failed,
        then (NaN, NaN) will be returned.
    """
    if isinstance(data, u.Quantity):
        data = data.value

    if np.issubdtype(data.dtype, int):
        # multiple code paths below require data be of float type
        # proactively convert int to float once and for all.
        data = data.astype(float)

    # Step 1: identify the patch of 3x3 pixels (z_)
    # that is centered on the brightest pixel (xx, yy)
    if mask is not None:
        # mask handling.
        # Issue 1401 demonstrates that using 'data' to find the max will break when all flux is negative
        # set masked pixels NaN (instead of 0) to resolve it.
        data = data.copy()
        data[~mask] = np.nan
    arg_data_max = np.nanargmax(data)
    yy, xx = np.unravel_index(arg_data_max, data.shape)
    # Make sure the 3x3 patch does not leave the TPF bounds
    if yy < 1:
        yy = 1
    if xx < 1:
        xx = 1
    if yy > (data.shape[0] - 2):
        yy = data.shape[0] - 2
    if xx > (data.shape[1] - 2):
        xx = data.shape[1] - 2

    z_ = data[yy - 1 : yy + 2, xx - 1 : xx + 2]
    if np.any(np.isnan(z_)):
        # handle edge case the 3X3 patch has NaN
        # Need some finite value for NaN pixels for the
        # quadratic fit below: use the mean of the 3x3 patch
        # to reduce the skew
        z_ = z_.copy()
        z_[np.isnan(z_)] = np.nanmean(z_)

    # Next, we will fit the coefficients of the bivariate quadratic with the
    # help of a design matrix (A) as defined by Eqn 20 in Vakili & Hogg
    # (arxiv:1610.05873). The design matrix contains a
    # column of ones followed by pixel coordinates: x, y, x**2, xy, y**2.
    A = np.array(
        [
            [1, -1, -1, 1, 1, 1],
            [1, 0, -1, 0, 0, 1],
            [1, 1, -1, 1, -1, 1],
            [1, -1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, -1, 1, 1, -1, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    # We also pre-compute $(A^t A)^-1 A^t$, cf. Eqn 21 in Vakili & Hogg.
    At = A.transpose()
    # In Python 3 this can become `Aprime = np.linalg.inv(At @ A) @ At`
    Aprime = np.matmul(np.linalg.inv(np.matmul(At, A)), At)

    # Step 2: fit the polynomial $P = a + bx + cy + dx^2 + exy + fy^2$
    # following Equation 21 in Vakili & Hogg.
    # In Python 3 this can become `Aprime @ z_.flatten()`
    a, b, c, d, e, f = np.matmul(Aprime, z_.flatten())

    # Step 3: analytically find the function maximum,
    # following https://en.wikipedia.org/wiki/Quadratic_function
    det = 4 * d * f - e ** 2
    if abs(det) < 1e-6:
        return np.nan, np.nan  # No solution
    xm = -(2 * f * b - c * e) / det
    ym = -(2 * d * c - b * e) / det
    return xx + xm, yy + ym


def _query_solar_system_objects(
    ra, dec, times, radius=0.1, location="kepler", cache=True, show_progress=True
):
    """Returns a list of asteroids/comets given a position and time.

    This function relies on The Virtual Observatory Sky Body Tracker (SkyBot)
    service which can be found at http://vo.imcce.fr/webservices/skybot/

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    times : array of float
        Times in Julian Date.
    radius : float
        Search radius in degrees.
    location : str
        Spacecraft location. Options include `'kepler'` and `'tess'`.
    cache : bool
        Whether to cache the search result. Default is True.
    show_progress : bool
        Whether to display a progress bar during the download. Default is True.

    Returns
    -------
    result : `pandas.DataFrame`
        DataFrame containing the list of known solar system objects at the
        requested time and location.
    """
    # We import pandas locally, because it takes quite a bit of time to import,
    # and it is only required for this specific feature.
    import pandas as pd

    if (location.lower() == "kepler") or (location.lower() == "k2"):
        location = "C55"
    elif location.lower() == "tess":
        location = "C57"

    url = "http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?"
    url += "-mime=text&"
    url += "-ra={}&".format(ra)
    url += "-dec={}&".format(dec)
    url += "-bd={}&".format(radius)
    url += "-loc={}&".format(location)

    df = None
    times = np.atleast_1d(times)
    for time in tqdm(times, desc="Querying for SSOs", disable=~show_progress):
        url_queried = url + "EPOCH={}".format(time)
        response = download_file(url_queried, cache=cache, show_progress=show_progress)
        if open(response).read(10) == "# Flag: -1":  # error code detected?
            raise IOError(
                "SkyBot Solar System query failed.\n"
                "URL used:\n" + url_queried + "\n"
                "Response received:\n" + open(response).read()
            )
        res = pd.read_csv(response, delimiter="|", skiprows=2)
        if len(res) > 0:
            res["epoch"] = time
            res.rename(
                {"# Num ": "Num", " Name ": "Name", " Class ": "Class", " Mv ": "Mv"},
                inplace=True,
                axis="columns",
            )
            res = res[["Num", "Name", "Class", "Mv", "epoch"]].reset_index(drop=True)
            if df is None:
                df = res
            else:
                df = pd.concat([df, res])
    if df is not None:
        df.reset_index(drop=True)
    return df


def show_citation_instructions():
    """Show citation instructions."""
    from . import PACKAGEDIR, __citation__

    # To make installing Lightkurve easier, ipython is an optional dependency,
    # because we can assume it is installed when notebook-specific features are called
    try:
        from IPython.display import HTML

        ipython_installed = True
    except ModuleNotFoundError:
        ipython_installed = False

    if not is_notebook() or not ipython_installed:
        print(__citation__)
    else:
        from pathlib import Path  # local import to speed up `import lightkurve`
        import astroquery  # local import to speed up `import lightkurve`

        templatefile = Path(PACKAGEDIR, "data", "show_citation_instructions.html")
        template = open(templatefile, "r").read()
        template = template.replace("LIGHTKURVE_CITATION", __citation__)
        template = template.replace("ASTROPY_CITATION", astropy.__citation__)
        template = template.replace("ASTROQUERY_CITATION", astroquery.__citation__)
        return HTML(template)


def _get_notebook_environment():
    """Returns 'jupyter', 'colab', or 'terminal'.

    One can detect whether or not a piece of Python is running by executing
    `get_ipython().__class__`, which returns the following result:

        * Jupyter notebook: `ipykernel.zmqshell.ZMQInteractiveShell`
        * Google colab: `google.colab._shell.Shell`
        * IPython terminal: `IPython.terminal.interactiveshell.TerminalInteractiveShell`
        * Python terminal: `NameError: name 'get_ipython' is not defined`
    """
    try:
        ipy = str(type(get_ipython())).lower()
        if "zmqshell" in ipy:
            return "jupyter"
        if "colab" in ipy:
            return "colab"
    except NameError:
        pass  # get_ipython() is not a builtin
    return "terminal"


def is_notebook():
    """Returns `True` if we are running in a notebook."""
    return _get_notebook_environment() in ["jupyter", "colab"]


def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.    If port is None we're asking about the URL
    for the origin header.
    """
    base_url = os.environ['LK_JUPYTERHUB_EXTERNAL_URL']
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    proxy_url_path = 'proxy/%d' % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


def finalize_notebook_url(notebook_url):
    """Based on `notebook_url` and the environment, compute a final value for
    notebook_url to be passed on to bokeh enabling transparent operation on JupyterHub.

    See Bokeh instructions here:
    https://docs.bokeh.org/en/latest/docs/user_guide/output/jupyter.html

    This handles two aspects of Bokeh made tricky by JupyterHub, firstly
    accessing the random Bokeh server port while behind a proxy, and second not
    triggering CORS restrictions while accessing a second server.

    A key aspect of the computed URL is the externally visible DNS name of the
    JupyterHub, so for the case of TIKE we might have:

    export LK_JUPYTERHUB_EXTERNAL_URL="https://timeseries.science.stsci.edu"

    If LK_JUPYTERHUB_EXTERNAL_URL is implicitly defined by the hub environment,
    JupyterHub users can nominally ignore the notebook_url parameter and
    Lightkurve should "just work" as if the local default URL localhost:8888
    was sufficient.

    For example remote_jupyter_proxy_url(25346) would return the URL
    "https://test.timeseries.science.stsci.edu/user/homer@stsci.edu/proxy/24356"

    which is essentially HUB + USER_SESSION + BOKEH_PORT_IN_SESSION

    The function result should be identical to past behavior unless the definition
    of LK_JUPYTERHUB_EXTERNAL_URL indicates JupyterHub is in use.  In this case the
    use of remote_jupyter_proxy_url is activated.   This effectively makes it the
    JupyterHub default instead of localhost:8888.
    """
    if notebook_url is not None:
        return notebook_url
    elif os.environ.get("LK_JUPYTERHUB_EXTERNAL_URL"):
        return remote_jupyter_proxy_url
    else:
        return "localhost:8888"
