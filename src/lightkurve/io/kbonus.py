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
        quarter_map = {
            ext.header["quarter"]: idx
            for idx, ext in enumerate(hdulist)
            if "QUARTER" in ext.header
        }
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
    with fits.open(lc.filename, lazy_load_hdus=True) as hdu:
        lc.meta["FLFRCSAP"] = hdu[2].header["FLFRCSAP"]
        lc.meta["CROWDSAP"] = hdu[2].header["CROWDSAP"]
    return KeplerLightCurve(data=lc, **kwargs)


def read_kbonus_lightcurve(filename, **kwargs):
    """Read a KBONUS-BKG light curve file.

    More information: https://archive.stsci.edu/hlsp/kbonus-bkg

    By default this will return the corrected, stitched light curve.
    Use the `quarters` argument to extract specific separated quarters.

    Parameters
    ----------
    filename : str
        Path or URL of a KBONUS-BKG light curve FITS file.
    quarters : None, int, list of int, or 'all'
        Which quarters to load. If None, will load the stitched light curve
        from the first extension. If a list of ints, int, or 'any', will load
        each of those quarters into a LightCurveCollection.

    Returns
    -------
    lc : `KeplerLightCurve`
    """

    _warn_bonus_file(filename)
    return _read_kbonus_lightcurve(filename, ext=1, **kwargs)


def read_kbonus_lightcurve_quarters(filename, quarters="any", **kwargs):
    """Read a KBONUS-BKG light curve file.

    More information: https://archive.stsci.edu/hlsp/kbonus-bkg

    By default this will return the corrected, stitched light curve.
    Use the `quarters` argument to extract specific separated quarters.

    Parameters
    ----------
    filename : str
        Path or URL of a KBONUS-BKG light curve FITS file.
    quarters : None, int, list of int, or 'all'
        Which quarters to load. If None, will load the stitched light curve
        from the first extension. If a list of ints, int, or 'any', will load
        each of those quarters into a LightCurveCollection.

    Returns
    -------
    lc : `LightCurveCollection`

    """
    _warn_bonus_file(filename)
    with fits.open(filename, lazy_load_hdus=True) as hdulist:
        quarter_map = {
            ext.header["quarter"]: idx
            for idx, ext in enumerate(hdulist)
            if "QUARTER" in ext.header
        }
    if isinstance(quarters, str):
        if quarters.lower() in ["all", "any"]:
            quarters = np.arange(18)
    if isinstance(quarters, (int, float, list, np.ndarray)):
        quarters = np.atleast_1d(quarters).astype(int)
        lcs = [
            _read_kbonus_lightcurve(filename, ext=quarter_map[quarter], **kwargs)
            for quarter in quarters
            if quarter in quarter_map
        ]
        return LightCurveCollection(lcs)
    else:
        raise ValueError("Can not parse `quarters`.")
