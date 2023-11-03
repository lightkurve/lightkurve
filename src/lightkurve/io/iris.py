"""Reader function for IRIS community light curve products."""
from ..lightcurve import KeplerLightCurve
from .generic import read_generic_lightcurve
import astropy.units as u
import numpy as np


def read_iris_lightcurve(filename, **kwargs):
    """Read a IRIS light curve file.

    More information: https://archive.stsci.edu/hlsp/iris

    Parameters
    ----------
    filename : str
        Path or URL of an IRIS light curve FITS file.

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    if "stitched" in filename:
        lc = read_generic_lightcurve(
            filename,
            flux_column="corrected_flux",
            time_format="bkjd",
            ext=1,
            cadenceno_column="CADENCE",
        )
        lc["flux"] = np.asarray(lc.flux.value) * u.Quantity(1, "")
        lc["flux_err"] = np.asarray(lc.flux_err.value) * u.Quantity(1, "")
        lc["flux"] += 1
        lc.meta["NORMALIZED"] = True
    else:
        lc = read_generic_lightcurve(
            filename,
            flux_column="flux",
            time_format="bkjd",
            ext=1,
            cadenceno_column="CADENCE",
        )

    lc.meta["AUTHOR"] = "IRIS"
    lc.meta["TARGETID"] = lc.meta.get("HLSPTARG")

    return KeplerLightCurve(data=lc, **kwargs)
