"""Reader function for K2SFF community light curve products."""
from astropy.io import fits

from .. import KeplerLightCurve
from ..utils import validate_method


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
    f = fits.open(filename)

    # Raise an exception if the requested extension is invalid
    validate_method(ext, supported_methods=[hdu.name.lower() for hdu in f])

    args = {
        "time": f[ext].data['T'],
        "flux": f[ext].data['FCOR'],
        "flux_unit": "",  # SFF light curves are normalized
        "cadenceno": f[ext].data['CADENCENO'],
        "targetid": f[0].header["KEPLERID"],
        "channel": f[0].header["CHANNEL"],
        "campaign": f[0].header["CAMPAIGN"],
        "mission": f[0].header["MISSION"],
        "ra": f[0].header["RA_OBJ"],
        "dec": f[0].header["DEC_OBJ"],
        "label": '{} (K2SFF)'.format(f[0].header["OBJECT"])
    }
    return KeplerLightCurve(**args)
