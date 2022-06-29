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
    return KeplerLightCurve(data=lc)
