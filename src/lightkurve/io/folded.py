"""Reader for official TESS light curve FITS files produced by the Ames SPOC pipeline."""
from ..lightcurve import FoldedLightCurve
#from ..utils import TessQualityFlags
from astropy.io import fits

from .generic import read_generic_lightcurve


def read_folded_lightcurve(
    filename, time_format="jd",
):
    """Returns a `~lightkurve.lightcurve.FoldedLightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of a TESS light curve FITS file.
    flux_column : 'pdcsap_flux' or 'sap_flux'
        Which column in the FITS file contains the preferred flux data?

    """


    lc = read_generic_lightcurve(filename, flux_column='FLUX', time_format='jd')
    hdu = fits.open(lc.filename)

    lc.meta["PERIOD"] = hdu[0].header["PERIOD"]
    lc.meta["NORMALIZE_PHASE"] = hdu[0].header["PH_NORM"]
    lc.meta["EPOCH_TIME"] = hdu[0].header["EPOCH"]
    lc.meta["EPOCH_PHASE"] = hdu[0].header["PH_EPOCH"]

    return FoldedLightCurve(data=lc)
