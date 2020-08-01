"""Functions for reading light curve data."""
import logging

from astropy.io import fits
from astropy.utils import deprecated

from .detect import detect_filetype
from ..lightcurve import KeplerLightCurve, TessLightCurve
from ..utils import LightkurveDeprecationWarning, LightkurveError

log = logging.getLogger(__name__)


__all__ = ['open', 'read']


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
    `~lightkurve.targetpixelfile.TargetPixelFile`.

    This function will use the `detect_filetype()` function to
    automatically detect the type of the data product, and return the
    appropriate object. File types currently supported include::

        * `KeplerTargetPixelFile` (typical suffix "-targ.fits.gz");
        * `KeplerLightCurve` (typical suffix "llc.fits");
        * `TessTargetPixelFile` (typical suffix "_tp.fits");
        * `TessLightCurve` (typical suffix "_lc.fits").

    Parameters
    ----------
    path_or_url : str
        Path or URL of a FITS file.

    Returns
    -------
    data : a subclass of  `~lightkurve.targetpixelfile.TargetPixelFile` or
        `~lightkurve.lightcurve.LightCurve`, depending on the detected file type.

    Raises
    ------
    ValueError : raised if the data product is not recognized as a Kepler or
        TESS product.

    Examples
    --------
    To read a target pixel file using its path or URL, simply use:

        >>> tpf = read("mytpf.fits")  # doctest: +SKIP
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
        if 'No such file' in str(e):
            raise e

    if filetype == "KeplerLightCurve":
        return KeplerLightCurve.read(path_or_url, format='kepler', **kwargs)
    elif filetype == "TessLightCurve":
        return TessLightCurve.read(path_or_url, format='tess', **kwargs)
    elif filetype == "K2SFF":
        return KeplerLightCurve.read(path_or_url, format='k2sff', **kwargs)
    elif filetype == "EVEREST":
        return KeplerLightCurve.read(path_or_url, format='everest', **kwargs)

    # Official data products;
    # if the filetype is recognized, instantiate a class of that name
    if filetype is not None:
        try:
            return getattr(__import__('lightkurve'), filetype)(path_or_url, **kwargs)
        except AttributeError as exc:
            raise LightkurveError(f"{filetype} files are not supported "
                                   "in this version of Lightkurve.") from exc
    else:
        # if these keywords don't exist, raise `ValueError`
        raise LightkurveError("Not recognized as a supported data product:\n"
                              f"{path_or_url}\n"
                              "This file may be corrupt due to an interrupted download. "
                              "Please remove it from your disk and try again.")
