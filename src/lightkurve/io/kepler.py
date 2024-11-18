"""Reader for official Kepler light curve FITS files produced by the Ames pipeline."""
from ..lightcurve import KeplerLightCurve
from ..utils import KeplerQualityFlags

from .generic import read_generic_lightcurve


def read_kepler_lightcurve(
    filename, flux_column="pdcsap_flux", quality_bitmask="default"
):
    """Returns a Kepler `~lightkurve.lightcurve.LightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of a Kepler light curve FITS file.
    flux_column : 'pdcsap_flux' or 'sap_flux'
        Which column in the FITS file contains the preferred flux data?
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=1130799`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=1664431`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=2096639`). This mask is not recommended.

        See the `~lightkurve.utils.KeplerQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(
        filename,
        flux_column=flux_column,
        quality_column="sap_quality",
        time_format="bkjd",
    )

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = KeplerQualityFlags.create_quality_mask(
        quality_array=lc["sap_quality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "Kepler"
    lc.meta["TARGETID"] = lc.meta.get("KEPLERID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask
    lc.meta["EXPTIME"] = lc.meta.get("FRAMETIM") * lc.meta.get("NUM_FRM")
    lc.meta["CADENCE_TYPE"] = get_kepler_cadence_type(lc.meta["EXPTIME"])
    return KeplerLightCurve(data=lc)

def get_kepler_cadence_type(exptime):
    """ Returns the Kepler cadence type as a string based on the exposure time.

    Alternatively, if a string is pass then this function checks for a valid 
    cadence type and then returns it. It raises an exception if not valid.

    The options are:
    if   exptime < 120   : "short"  (i.e. 60-second)
    elif exptime < 2000 : "long" (i.e. 30-minute)
    else                : "ffi"  (i.e. Full-Frame Images)

    Parameters
    ----------
    exptime : 'ffi', 'long', 'short', or float
        Exposure time for cadence in seconds
        Or a string containing the cadence type
        'ffi' selects FFI products.
        'long' selects 30-min cadence products;
        'short' selects 1-min products;

    Returns
    -------
    cadence_type : str
        one of: ('ffi', 'long', 'short')
    """
    valid_str_options = ('ffi', 'long', 'short')

    if isinstance(exptime, str):
        if valid_str_options.count(exptime.lower()) != 1:
            raise Exception('Invalid cadence type')
        else:
            return exptime

    if exptime is None:
        return None
    elif exptime < 120:
        return 'short'
    elif exptime < 2000:
        return 'long'
    else:
        return 'ffi'
