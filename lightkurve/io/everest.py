"""Reader for K2 EVEREST light curves."""
from astropy.io import fits

from .. import KeplerLightCurve


def read_everest_lightcurve(path_or_url, **kwargs):
    """Read an EVEREST light curve file.
    More information: https://archive.stsci.edu/hlsp/everest
    Parameters
    ----------
    path_or_url : str
        Path or URL of a K2SFF light curve FITS file.
    ext : str
        Version of the light curve to use.  Valid options include "BESTAPER",
        "CIRC_APER0" through "CIRC_APER9", and "PRF_APER0" through "PRF_APER9".
    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    f = fits.open(path_or_url)

    args = {
        "time": f[1].data['TIME'],
        "flux": f[1].data['FCOR'],
        "flux_err": f[1].data['FRAW_ERR'],
        "cadenceno": f[1].data['CADN'],
        "quality": f[1].data['QUALITY'],
        "targetid": f[0].header["KEPLERID"],
        "channel": f[0].header["CHANNEL"],
        "campaign": f[0].header["CAMPAIGN"],
        "mission": f[0].header["MISSION"],
        "ra": f[0].header["RA_OBJ"],
        "dec": f[0].header["DEC_OBJ"],
        "label": '{} (EVEREST)'.format(f[0].header["OBJECT"])
    }
    return KeplerLightCurve(**args)
