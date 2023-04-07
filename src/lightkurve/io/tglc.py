"""Reader for TGLC light curve files.
Details can be found at https://archive.stsci.edu/hlsp/tglc
"""
import numpy as np
from astropy import units as u

from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags
from .generic import read_generic_lightcurve


def read_tglc_lightcurve(
    filename, flux_column="cal_psf_flux", quality_bitmask="default"
):
    """Returns a `~lightkurve.lightcurve.LightCurve` object given a light curve file from
    TGLC HLSP

    By default, TGLC's `cal_psf_flux` values are used for the `flux` column. No errors are
    provided by this HLSP.

    Note this reader does not use the TGLC_FLAG extension to inform the bitmask for the
    returned light curve, but those flags can still be accessed.

    More information on TGLC: https://archive.stsci.edu/hlsp/tglc

    Parameters
    ----------
    filename : str
        Local path or remote url of a TGLC light curve FITS file.
    flux_column : 'CAL_PSF_FLUX', 'CAL_APER_FLUX', 'PSF_FLUX', or 'APERTURE_FLUX'
        Which column in the FITS file contains the preferred flux data?
        By default the "Corrected PSF Flux" flux (CAL_PSF_FLUX) is used.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:
            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with flags indicating AttitudeTweak, SafeMode, CoarsePoint, EarthPoint, Desat, or
              ManualExclude will be ignored.
            * "hard": cadences with default flags, ApertureCosmic, CollateralCosmic, Straylight, or Straylight2 will be
              ignored.
            * "hardest": cadences with all the above flags will be ignored, in addition to cadences with GSFC-ELEANOR-LITE
              bit flags of 17 (decimal value 131072) and 18 (decimal value 262144).
    """

    lc = read_generic_lightcurve(
        filename,
        time_column="time",
        flux_column=flux_column.lower(),
        quality_column="tess_flags",
        cadenceno_column="cadence_num",
        time_format="btjd",
    )

    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )

    # TGLC FITS file do not have units specified. re-add them.
    for colname in ["psf_flux", "aperture_flux", "background"]:
        if colname in lc.colnames:
            if lc[colname].unit is not None:
                # for case flux, flux_err, lightkurve has forced it to be u.dimensionless_unscaled
                # can't reset a unit, so we create a new column
                lc[colname] = u.Quantity(
                    lc[colname].value, "electron/s", dtype=np.float32
                )
            else:
                lc[colname].unit = "electron/s"

    # Calibrated columns are normalized, so they are unitless
    for colname in ["cal_psf_flux", "cal_aper_flux"]:
        if colname in lc.colnames:
            if lc[colname].unit is not None:
                # for case flux, flux_err, lightkurve has forced it to be u.dimensionless_unscaled
                # can't reset a unit, so we create a new column
                lc[colname] = u.Quantity(lc[colname].value, "", dtype=np.float32)
            else:
                lc[colname].unit = ""

    lc = lc[quality_mask]
    lc.meta["AUTHOR"] = "TGLC"
    lc.meta["TARGETID"] = lc.meta.get("OBJECT")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask
    lc.meta["NORMALIZED"] = True
    tic = lc.meta.get("TICID")
    if tic is not None:
        tic = int(tic)
        # compatibility with SPOC, QLP, etc.
        lc.meta["TARGETID"] = tic
        lc.meta["TICID"] = tic
        lc.meta["OBJECT"] = f"TIC {tic}"
        # for Lightkurve's plotting methods
        lc.meta["LABEL"] = f"TIC {tic}"
    return TessLightCurve(data=lc)
