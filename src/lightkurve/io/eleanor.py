"""Reader for ELEANOR light curve files.

Website: https://github.com/afeinstein20/eleanor

It is used by HLSP GSFC-ELEANOR-LITE.

Website: https://archive.stsci.edu/hlsp/gsfc-eleanor-lite
"""
from astropy import units as u
from matplotlib.pyplot import tick_params
import numpy as np

from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_eleanor_lightcurve(filename, flux_column="corr_flux", quality_bitmask=None):
    """Returns a `TessLightCurve` object given a light curve file from the ELEANOR.

    By default, ELEANOR's `corr_flux` column is used to populate the `flux` values,
    with no flux error (corr_flux does not come with error).

    Parameters
    ----------
    filename : str
        Local path or remote url of a Eleanor light curve FITS file.
    flux_column : 'raw_flux', 'corr_flux', 'pca_flux', "psf_flux', or 'flux_bkg'
        Which column in the FITS file contains the preferred flux data?
        By default the corrected flux (corr_flux) is used.
    quality_bitmask:
        ignored. It is retained to be compatible with generic read interface.
    """

    # See eleanor's TargetData.save() for how the files are created
    # https://adina.feinste.in/eleanor/api.html#eleanor.TargetData.save
    # A recent version:
    # https://github.com/afeinstein20/eleanor/blob/cf19a998a99d0cea09c8426f8db2248fdc55f12c/eleanor/targetdata.py#L1366

    lc = read_generic_lightcurve(filename, flux_column=flux_column, flux_err_column=None, time_format="btjd")

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

    for colname in ["barycorr"]:
        if colname in lc.colnames:
            lc[colname].unit = u.day

    # OPEN: x_centroid	y_centroid	x_com	y_com

    # In Eleanor fits file, raw_flux's error is in flux_err, which breaks Lightkurve convention.
    # Fix it here
    lc["raw_flux_err"] = lc["flux_err"]
    if flux_column.lower() != 'raw_flux':  #
        lc["flux_err"] = np.full_like(lc["flux_err"], np.nan)

    if lc.meta.get("AUTHOR") is not None:
        # the author header is populated with "Adina D. Feinstein", which is not as useful in Lightkurve context
        lc.meta["AUTHOR_ELEANOR"] = lc.meta.get("AUTHOR")
    lc.meta["AUTHOR"] = "ELEANOR"

    tic = lc.meta.get("TIC_ID")
    if tic is not None:
        # compatibility with SPOC, QLP, etc.
        lc.meta["TARGETID"] = tic
        lc.meta["TICID"] = tic
        lc.meta["OBJECT"] = f"TIC {tic}"
        # for Lightkurve's plotting methods
        lc.meta["LABEL"] = f"TIC {tic}"

    return TessLightCurve(data=lc)
