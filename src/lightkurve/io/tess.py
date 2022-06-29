"""Reader for official TESS light curve FITS files produced by the Ames SPOC pipeline."""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_tess_lightcurve(
    filename, flux_column="pdcsap_flux", quality_bitmask="default"
):
    """Returns a TESS `~lightkurve.lightcurve.LightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of a TESS light curve FITS file.
    flux_column : 'pdcsap_flux' or 'sap_flux'
        Which column in the FITS file contains the preferred flux data?
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored.
            * "default": cadences with severe quality issues will be ignored.
            * "hard": more conservative choice of flags to ignore.
              This is known to remove good data.
            * "hardest": removes all data that has been flagged.
              This mask is not recommended.

        See the `~lightkurve.utils.TessQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(filename, flux_column=flux_column, time_format="btjd")

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "SPOC"
    lc.meta["TARGETID"] = lc.meta.get("TICID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask
    return TessLightCurve(data=lc)
