"""Reader for GSFC-ELEANOR-LITE light curve files. 
Details can be found at https://archive.stsci.edu/hlsp/eleanor and https://archive.stsci.edu/hlsp/gsfc-eleanor-lite
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
    quality_bitmask=None, 
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
      Please note that the "FLUX_ERR" column in Eleanor FITS file is referred to the uncertainty of "RAW_FLUX", not "CORR_FLUX"
    quality_bitmask : Not used
    """
    lc = read_generic_lightcurve(
        filename,
        time_column=time_column.lower(),
        flux_column=flux_column.lower(),
        flux_err_column = flux_err_column.lower(),
        time_format="btjd",
        quality_column= quality_column.lower(),
        centroid_col_column = centroid_col_column.lower(),
        centroid_row_column = centroid_row_column.lower(),
        cadenceno_column = cadenceno_column.lower()
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
