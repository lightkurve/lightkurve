"""Reader function for KBONUS-BKG community light curve products."""
from ..lightcurve import KeplerLightCurve
from ..collections import LightCurveCollection
from .generic import read_generic_lightcurve
from astropy.io import fits
import numpy as np

import logging

log = logging.getLogger(__name__)


def _warn_bonus_file(filename):
    with fits.open(filename, lazy_load_hdus=True) as hdulist:
        if hdulist[0].header["KEPMAG"] < 12:
            log.warning(
                "Kepler magnitude is bright (less than 12), indicating the target is saturated. KBONUS-BKG data is invalid for saturated targets."
            )
        psffrac = np.asarray(
            [
                ext.header["PSFFRAC"]
                for idx, ext in enumerate(hdulist)
                if "QUARTER" in ext.header
            ]
        )
        if (psffrac < 0.6).any():
            log.warning(
                "PSF Fraction is low, indicating not all of the flux is captured in the PSF model. This data may be unreliable."
            )
    return


def _read_kbonus_lightcurve(filename, ext, **kwargs):
    """Read an extension of kbonus light curve"""
    lc = read_generic_lightcurve(
        filename,
        time_format="bkjd",
        ext=ext,
        centroid_col_column="centroid_column",
        centroid_row_column="centroid_row",
    )

    lc.meta["AUTHOR"] = "KBONUS-BKG"
    lc.meta["TARGETID"] = lc.meta.get("GAIAID")
    if ext != 1:
        with fits.open(lc.filename, lazy_load_hdus=True) as hdulist:
            lc.meta["FLFRCSAP"] = hdulist[ext].header["FLFRCSAP"]
            lc.meta["CROWDSAP"] = hdulist[ext].header["CROWDSAP"]
    return KeplerLightCurve(data=lc, **kwargs)


def read_kbonus_lightcurve(filename, quarter=None, **kwargs):
    """Read a KBONUS-BKG light curve file.

    More information: https://archive.stsci.edu/hlsp/kbonus-bkg

    By default this will return the corrected, stitched light curve.
    Use the `quarter` argument to extract specific separated quarter.

    Parameters
    ----------
    filename : str
        Path or URL of a KBONUS-BKG light curve FITS file.
    quarter : None or int
        Which quarter to load. If None, will load the stitched light curve
        from the first extension. If an int, will load that quarter only.

    Returns
    -------
    lc : `LightCurveCollection`

    """
    _warn_bonus_file(filename)
    if quarter is None:
        return _read_kbonus_lightcurve(filename, ext=1, **kwargs)
    with fits.open(filename, lazy_load_hdus=True) as hdulist:
        quarter_map = {
            ext.header["quarter"]: idx
            for idx, ext in enumerate(hdulist)
            if "QUARTER" in ext.header
        }
        gaiaid = hdulist[0].header["OBJECT"]

    if isinstance(quarter, str):
        if quarter.lower() in ["all", "any"]:
            quarter = None
    if isinstance(quarter, int):
        try:
            lc = _read_kbonus_lightcurve(filename, ext=quarter_map[quarter], **kwargs)
        except KeyError:
            try:
                lc = _read_kbonus_lightcurve(filename, ext=2, **kwargs)[:0]
            except KeyError:
                lc = _read_kbonus_lightcurve(filename, ext=1, **kwargs)[:0]
            lc.meta = {
                "QUARTER": quarter,
                "AUTHOR": "KBONUS-BKG",
                "GAIAID": gaiaid,
                "LABEL": gaiaid,
            }

        lc.meta["TARGETID"] = lc.meta.get("GAIAID")
        return lc
    else:
        raise ValueError("Can not parse `quarter`.")
