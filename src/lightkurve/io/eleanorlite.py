"""Reader for GSFC-ELEANOR-LITE light curve files. 
Details can be found at https://archive.stsci.edu/hlsp/eleanor and https://archive.stsci.edu/hlsp/gsfc-eleanor-lite
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags
from astropy import units as u

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

    By default, eleanor's `CORR_FLUX` column is used to populate the `flux` values. Note that the "FLUX_ERR" column in the 
    Eleanor FITS file is referred to the uncertainty of "RAW_FLUX", not "CORR_FLUX". Thus the uncertainty reported in the 'flux_err' column here is calculated as follows: corr_flux_err = corr_flux*raw_flux_err/raw_flux. For completeness, the original raw_flux's error is added as a "raw_flux_err" column.

    Parameters
    ----------
    filename : str
        Local path or remote url of a QLP light curve FITS file.
    flux_column : 'RAW_FLUX', 'CORR_FLUX', 'PCA_FLUX', or 'FLUX_BKG'
        Which column in the FITS file contains the preferred flux data?
        By default the "Corrected Flux" flux (CORR_FLUX) is used.
    flux_err_column: 'FLUX_ERR'
      Which column in the FITS file contains the preferred flux_err data?
      The corr_flux error is calculated from corr_flux_err = corr_flux*raw_flux_err/raw_flux. For completeness, the original raw_flux's error is added as a "raw_flux_err" column
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

    # Eleanor FITS file do not have units specified. re-add them.
    for colname in ["flux", "flux_err", "raw_flux", "corr_flux", "pca_flux", "psf_flux"]:
        if colname in lc.colnames:
            if lc[colname].unit is not None:
                # for case flux, flux_err, lightkurve has forced it to be u.dimensionless_unscaled
                # can't reset a unit, so we create a new column
                lc[colname] = u.Quantity(lc[colname].value, "electron/s")
            else:
                lc[colname].unit = "electron/s"

    for colname in ["flux_bkg"]:
        if colname in lc.colnames:
            lc[colname].unit = u.percent

    for colname in ["centroid_col", "centroid_row", "x_centroid", "y_centroid", "x_com", "y_com"]:
        if colname in lc.colnames:
            lc[colname].unit = u.pix

    for colname in ["barycorr"]:
        if colname in lc.colnames:
            lc[colname].unit = u.day

    # In Eleanor fits file, raw_flux's error is in flux_err, which breaks Lightkurve convention.
    # To account for this, the corr_flux error is calculated from corr_flux_err = corr_flux*raw_flux_err/raw_flux. For completeness, 
    # the original raw_flux's error is added as a "raw_flux_err" column
    lc["raw_flux_err"] = lc["flux_err"]
    if flux_column.lower() != 'raw_flux':
        lc["flux_err"] = lc[flux_column.lower()]*lc["raw_flux_err"]/lc["raw_flux"]

    lc.meta["AUTHOR"] = "GSFC-ELEANOR-LITE"

    # Eleanor light curves are not normalized by default
    lc.meta["NORMALIZED"] = False

    tic = lc.meta.get("TIC_ID")
    if tic is not None:
        # compatibility with SPOC, QLP, etc.
        lc.meta["TARGETID"] = tic
        lc.meta["TICID"] = tic
        lc.meta["OBJECT"] = f"TIC {tic}"
        # for Lightkurve's plotting methods
        lc.meta["LABEL"] = f"TIC {tic}"


    return TessLightCurve(data=lc, **kwargs)
