"""Reader for folded light curve FITS files produced by lightkurve.lightcurve.FoldedLightCurve."""
from ..lightcurve import FoldedLightCurve
from astropy.io import fits

from .generic import read_generic_lightcurve


def read_folded_lightcurve(
    filename, time_format='jd',
):
    """
    Reads in lightcurves created by lightkurve.lightcurve.FoldedLightCurve().to_fits()
    Returns a `~lightkurve.lightcurve.FoldedLightCurve()`.

    Parameters
    ----------
    filename : str
        Local path or remote url of a TESS light curve FITS file.
    time_format : str
        Default 'jd' which assumes units of days. 
        If the folded time is normalized between 0 and 1, time will be converted to an astropy Quantity instead of a Time object

    """
    
    lc = read_generic_lightcurve(filename, flux_column='FLUX', time_format=time_format)

    if isinstance(lc.filename, fits.HDUList):
        hdu = lc.filename
    else: 
        hdu = fits.open(lc.filename)

        
    # These features are automatically added by lightkurve when creating a folded lc
    # They are required in the meta data for functions such as plotting
    lc.meta["PERIOD"] = hdu[0].header["PERIOD"]
    lc.meta["NORMALIZE_PHASE"] = hdu[0].header["PHNORM"]
    lc.meta["EPOCH_TIME"] = hdu[0].header["EPOCH"]
    lc.meta["EPOCH_PHASE"] = hdu[0].header["PHEPOCH"]

    flc = FoldedLightCurve(data=lc)
    if flc.normalize_phase == True:
        flc._restore_normalized_phase()
    return flc
