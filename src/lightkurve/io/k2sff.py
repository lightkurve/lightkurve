"""Reader function for K2SFF community light curve products."""
from ..lightcurve import KeplerLightCurve
from ..utils import validate_method

from .generic import read_generic_lightcurve


def read_k2sff_lightcurve(filename, ext="BESTAPER", **kwargs):
    """Read a K2SFF light curve file.

    More information: https://archive.stsci.edu/hlsp/k2sff

    Parameters
    ----------
    filename : str
        Path or URL of a K2SFF light curve FITS file.
    ext : str
        Version of the light curve to use.  Valid options include "BESTAPER",
        "CIRC_APER0" through "CIRC_APER9", and "PRF_APER0" through "PRF_APER9".

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    lc = read_generic_lightcurve(
        filename, flux_column="fcor", time_format="bkjd", ext=ext
    )

    lc.meta["AUTHOR"] = "K2SFF"
    lc.meta["TARGETID"] = lc.meta.get("KEPLERID")

    return KeplerLightCurve(data=lc, **kwargs)
