"""Reader function for KEPSEISMIC community light curve products."""
from ..lightcurve import KeplerLightCurve

from .generic import read_generic_lightcurve

def read_kepseismic_lightcurve(filename, **kwargs):
    """Read a KEPSEISMIC light curve file.

    More information: https://archive.stsci.edu/prepds/kepseismic

    Parameters
    ----------
    filename : str
        Path or URL of a K2SFF light curve FITS file.

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """

    lc = read_generic_lightcurve(
        filename,
        time_format='mjd')

    lc.meta["AUTHOR"] = "KEPSEISMIC"
    lc.meta["TARGETID"] = lc.meta.get("KEPLERID")

    # KEPSEISMIC light curves are normalized by default
    lc.meta["NORMALIZED"] = True

    return KeplerLightCurve(data=lc, **kwargs)
