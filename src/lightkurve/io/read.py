"""Functions for reading light curve data."""
import logging

from astropy.io import fits
from astropy.utils import deprecated

from ..lightcurve import KeplerLightCurve, TessLightCurve
from ..utils import LightkurveDeprecationWarning, LightkurveError
from .detect import detect_filetype

log = logging.getLogger(__name__)


__all__ = ["open", "read"]


@deprecated("2.0", alternative="read()", warning_type=LightkurveDeprecationWarning)
def open(path_or_url, **kwargs):
    """DEPRECATED. Please use `lk.read()` instead.

    This function has been deprecated because its name collides with Python's
    built-in `open()` function.
    """
    return read(path_or_url, **kwargs)


def read(path_or_url, **kwargs):
    """Reads any valid Kepler or TESS data file and returns an instance of
    `~lightkurve.lightcurve.LightCurve` or
    `TargetPixelFile <../targetpixelfile.html>`_

    This function will automatically detect the type of the data product, and return the
    appropriate object. File types currently supported include::

        * `KeplerTargetPixelFile` (typical suffix "-targ.fits.gz");
        * `KeplerLightCurve` (typical suffix "llc.fits");
        * `TessTargetPixelFile` (typical suffix "_tp.fits");
        * `TessLightCurve` (typical suffix "_lc.fits").

    Parameters
    ----------
    path_or_url : str
        Path or URL of a FITS file.
    quality_bitmask : str or int, optional
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored
            * "default": cadences with severe quality issues will be ignored
            * "hard": more conservative choice of flags to ignore
              This is known to remove good data.
            * "hardest": removes all data that has been flagged
              This mask is not recommended.

        See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` or :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
    flux_column : str, optional
        (Applicable to LightCurve products only) The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
        Typically 'pdcsap_flux' or 'sap_flux'.
    **kwargs : dict
        Dictionary of arguments to be passed to underlying data product type specific reader.

    Returns
    -------
    data : a subclass of `~lightkurve.lightcurve.LightCurve` or `TargetPixelFile <../targetpixelfile.html>`_
           depending on the detected file type.

    Raises
    ------
    ValueError : raised if the data product is not recognized as a Kepler or TESS product.

    Examples
    --------
    To read a target pixel file using its path or URL, simply use:

        >>> import lightkurve as lk
        >>> tpf = lk.read("mytpf.fits")  # doctest: +SKIP
    """
    log.debug("Opening {}.".format(path_or_url))
    # pass header into `detect_filetype()`
    try:
        with fits.open(path_or_url) as temp:
            filetype = detect_filetype(temp)
            log.debug("Detected filetype: '{}'.".format(filetype))
    except OSError as e:
        filetype = None
        # Raise an explicit FileNotFoundError if file not found
        if "No such file" in str(e):
            raise e

    try:
        if filetype == "KeplerLightCurve":
            return KeplerLightCurve.read(path_or_url, format="kepler", **kwargs)
        elif filetype == "TessLightCurve":
            return TessLightCurve.read(path_or_url, format="tess", **kwargs)
        elif filetype == "QLP":
            return TessLightCurve.read(path_or_url, format="qlp", **kwargs)
        elif filetype == "ELEANOR":
            return TessLightCurve.read(path_or_url, format="eleanor", **kwargs)
        elif filetype == "PATHOS":
            return TessLightCurve.read(path_or_url, format="pathos", **kwargs)
        elif filetype == "CDIPS":
            return TessLightCurve.read(path_or_url, format="cdips", **kwargs)
        elif filetype == "TASOC":
            return TessLightCurve.read(path_or_url, format="tasoc", **kwargs)
        elif filetype == "K2SFF":
            return KeplerLightCurve.read(path_or_url, format="k2sff", **kwargs)
        elif filetype == "EVEREST":
            return KeplerLightCurve.read(path_or_url, format="everest", **kwargs)
        elif filetype == "KEPSEISMIC":
            return KeplerLightCurve.read(path_or_url, format="kepseismic", **kwargs)
        elif filetype == "TGLC":
            return TessLightCurve.read(path_or_url, format="tglc", **kwargs)
    except BaseException as exc:
        # ensure path_or_url is in the error
        raise LightkurveError(
            f"Error in reading Data product {path_or_url} of type {filetype} .\n"
            "This file may be corrupt due to an interrupted download. "
            "Please remove it from your disk and try again."
        ) from exc

    # Official data products;
    # if the filetype is recognized, instantiate a class of that name
    if filetype is not None:
        try:
            return getattr(__import__("lightkurve"), filetype)(path_or_url, **kwargs)
        except AttributeError as exc:
            raise LightkurveError(
                f"Data product f{path_or_url} of type {filetype} is not supported "
                "in this version of Lightkurve."
            ) from exc
    else:
        # if these keywords don't exist, raise `ValueError`
        raise LightkurveError(
            "Not recognized as a supported data product:\n"
            f"{path_or_url}\n"
            "This file may be corrupt due to an interrupted download. "
            "Please remove it from your disk and try again."
        )
