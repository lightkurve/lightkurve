"""Functions for reading light curve data."""
import logging
import warnings

from astropy.io import fits

from .lightcurve import KeplerLightCurve
from .utils import validate_method, LightkurveWarning

log = logging.getLogger(__name__)

__all__ = ['open', 'read', 'read_k2sff', 'read_everest']


def open(path_or_url, **kwargs):
    """DEPRECATED. Please use `lk.read()` instead.

    This function has been deprecated because its name collides with Python's
    built-in `open()` function.
    """
    warnings.warn("`lightkurve.open()` is deprecated, please use "
                  "`lightkurve.read()` instead.",
                    LightkurveWarning)
    return read(path_or_url, **kwargs)


def read(path_or_url, **kwargs):
    """Reads any valid Kepler or TESS data file and returns an instance of
    `~lightkurve.lightcurvefile.LightCurveFile` or
    `~lightkurve.targetpixelfile.TargetPixelFile`.

    This function will use the `detect_filetype()` function to
    automatically detect the type of the data product, and return the
    appropriate object. File types currently supported are::

        * `KeplerTargetPixelFile` (typical suffix "-targ.fits.gz");
        * `KeplerLightCurveFile` (typical suffix "llc.fits");
        * `TessTargetPixelFile` (typical suffix "_tp.fits");
        * `TessLightCurveFile` (typical suffix "_lc.fits").

    Parameters
    ----------
    path_or_url : str
        Path or URL of a FITS file.

    Returns
    -------
    data : a subclass of  `~lightkurve.targetpixelfile.TargetPixelFile` or
        `~lightkurve.lightcurvefile.LightCurveFile`, depending on the detected file type.

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

    # Community-provided science products
    if filetype == "K2SFF":
        return read_k2sff(path_or_url, **kwargs)
    elif filetype == "EVEREST":
        return read_everest(path_or_url, **kwargs)

    # Official data products;
    # if the filetype is recognized, instantiate a class of that name
    if filetype is not None:
        return getattr(__import__('lightkurve'), filetype)(path_or_url, **kwargs)
    else:
        # if these keywords don't exist, raise `ValueError`
        raise ValueError("Not recognized as a Kepler or TESS data product: "
                         "{}".format(path_or_url))


def read_k2sff(path_or_url, ext="BESTAPER", **kwargs):
    """Read a K2SFF light curve file.

    More information: https://archive.stsci.edu/hlsp/k2sff

    Parameters
    ----------
    path_or_url : str
        Path or URL of a K2SFF light curve FITS file.
    ext : str
        Version of the light curve to use.  Valid options include "BESTAPER",
        "CIRC_APER0" through "CIRC_APER9", and "PRF_APER0" through "PRF_APER9".

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    f = fits.open(path_or_url)

    # Raise an exception if the requested extension is invalid
    validate_method(ext, supported_methods=[hdu.name.lower() for hdu in f])

    args = {
        "time": f[ext].data['T'],
        "flux": f[ext].data['FCOR'],
        "flux_unit": "",  # SFF light curves are normalized
        "cadenceno": f[ext].data['CADENCENO'],
        "targetid": f[0].header["KEPLERID"],
        "channel": f[0].header["CHANNEL"],
        "campaign": f[0].header["CAMPAIGN"],
        "mission": f[0].header["MISSION"],
        "ra": f[0].header["RA_OBJ"],
        "dec": f[0].header["DEC_OBJ"],
        "label": '{} (K2SFF)'.format(f[0].header["OBJECT"])
    }
    return KeplerLightCurve(**args)


def read_everest(path_or_url, **kwargs):
    """Read an EVEREST light curve file.

    More information: https://archive.stsci.edu/hlsp/everest

    Parameters
    ----------
    path_or_url : str
        Path or URL of a K2SFF light curve FITS file.
    ext : str
        Version of the light curve to use.  Valid options include "BESTAPER",
        "CIRC_APER0" through "CIRC_APER9", and "PRF_APER0" through "PRF_APER9".

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    f = fits.open(path_or_url)

    args = {
        "time": f[1].data['TIME'],
        "flux": f[1].data['FCOR'],
        "flux_err": f[1].data['FRAW_ERR'],
        "cadenceno": f[1].data['CADN'],
        "quality": f[1].data['QUALITY'],
        "targetid": f[0].header["KEPLERID"],
        "channel": f[0].header["CHANNEL"],
        "campaign": f[0].header["CAMPAIGN"],
        "mission": f[0].header["MISSION"],
        "ra": f[0].header["RA_OBJ"],
        "dec": f[0].header["DEC_OBJ"],
        "label": '{} (EVEREST)'.format(f[0].header["OBJECT"])
    }
    return KeplerLightCurve(**args)


def detect_filetype(hdulist):
    """Returns Kepler and TESS file types given a FITS object.

    This function will detect the file type by looking at both the TELESCOP and
    CREATOR keywords in the first extension of the FITS header. If the file is
    recognized as a Kepler or TESS data product, one of the following strings
    will be returned:

        * `'KeplerTargetPixelFile'`
        * `'TessTargetPixelFile'`
        * `'KeplerLightCurveFile'`
        * `'TessLightCurveFile'`

    In addition, community-provided data products such as K2SFF are supported.

    If the data product cannot be detected, `None` will be returned.

    Parameters
    ----------
    hdulist : astropy.io.fits.HDUList object
        A FITS file.

    Returns
    -------
    filetype : str or None
        A string describing the detected filetype. If the filetype is not
        recognized, `None` will be returned.
    """
    # Is it a K2SFF file?
    try:
        # There are no metadata keywords identifying K2SFF FITS files,
        # so we go by structure.
        if hdulist[1].header['EXTNAME'] == "BESTAPER" and hdulist[1].header["TTYPE4"] == "ARCLENGTH":
            return "K2SFF"
    except Exception:
        pass

    # Is it an EVEREST file?
    try:
        if "EVEREST" in str(hdulist[0].header['COMMENT']):
            return "EVEREST"
    except Exception:
        pass

    # Is it an official data product?
    header = hdulist[0].header
    try:
        # use `telescop` keyword to determine mission
        # and `creator` to determine tpf or lc
        if 'TELESCOP' in header.keys():
            telescop = header['telescop'].lower()
        else:
            # Some old custom TESS data did not define the `TELESCOP` card
            telescop = header['mission'].lower()
        creator = header['creator'].lower()
        origin = header['origin'].lower()
        if telescop == 'kepler':
            # Kepler TPFs will contain "TargetPixelExporterPipelineModule"
            if 'targetpixel' in creator:
                return 'KeplerTargetPixelFile'
            # Kepler LCFs will contain "FluxExporter2PipelineModule"
            elif ('fluxexporter' in creator or 'lightcurve' in creator
                or 'lightcurve' in creator):
                return 'KeplerLightCurveFile'
        elif telescop == 'tess':
            # TESS TPFs will contain "TargetPixelExporterPipelineModule"
            if 'targetpixel' in creator:
                return 'TessTargetPixelFile'
            # TESS LCFs will contain "LightCurveExporterPipelineModule"
            elif 'lightcurve' in creator:
                return 'TessLightCurveFile'
            # Early versions of TESScut did not set a good CREATOR keyword
            elif 'stsci' in origin:
                return 'TessTargetPixelFile'
    # If the TELESCOP or CREATOR keywords don't exist we expect a KeyError;
    # if one of them is Undefined we expect `.lower()` to yield an AttributeError.
    except (KeyError, AttributeError):
        return None
