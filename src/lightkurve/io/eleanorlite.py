"""Reader for GSFC-ELEANOR-LITE light curve files.
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve

def read_eleanorlite_lightcurve(filename,
    time_column="TIME",
    flux_column="CORR_FLUX", 
    flux_err_column="FLUX_ERR", 
    cadenceno_column="FFIINDEX",
    centroid_col_column="X_CENTROID",
    centroid_row_column="Y_CENTROID",
    quality_column="QUALITY",
    quality_bitmask="default", 
    **kwargs):
    """Returns a `TessLightCurve` object given a light curve file from the GSFC Eleanor-lite Pipeline.

    By default, eleanor's `CORR_FLUX` column is used to populate the `flux` values,
    and 'FLUX_ERR' is used to populate `flux_err`. 

    Parameters
    ----------
    filename : str
        Local path or remote url of a QLP light curve FITS file.
    flux_column : 'RAW_FLUX', 'CORR_FLUX', 'PCA_FLUX', or 'FLUX_BKG'
        Which column in the FITS file contains the preferred flux data?
        By default the "Corrected Flux" flux (CORR_FLUX) is used.
    flux_err_column: 'FLUX_ERR'
      Which column in the FITS file contains the preferred flux_err data?
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

        See the :class:`TessQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(
        filename,
        time_column=time_column.lower(),
        flux_column=flux_column.lower(),
        time_format="btjd",
        quality_column= quality_column.lower(),
    )

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "GSFC-ELEANOR-LITE"
    lc.meta["TARGETID"] = lc.meta.get("TIC_ID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask

    # Eleanor light curves are not normalized by default
    lc.meta["NORMALIZED"] = False

    return TessLightCurve(data=lc, **kwargs)
