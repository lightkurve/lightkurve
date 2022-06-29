"""Reader for A PSF-Based Approach to TESS Cluster Difference Imaging Photometric Survey (CDIPS) light curves

Website: https://archive.stsci.edu/hlsp/cdips
Product Description: https://archive.stsci.edu/hlsps/cdips/hlsp_cdips_tess_ffi_all_tess_v01_readme.md
"""
import logging

from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve

log = logging.getLogger(__name__)

def read_cdips_lightcurve(filename,
                            flux_column="IRM1",
                            include_inst_errs=False,
                            quality_bitmask=None):
    """Returns a TESS CDIPS `~lightkurve.lightcurve.LightCurve`.

    Note: CDIPS light curves have already had quality filtering applied, and
    do not provide the bitflags necessary for a user to apply a new bitmask.
    Therefore, frames corresponding to "momentum dumps and coarse point modes"
    are removed according to Bouma et al. 2019, and no other quality filtering
    is allowed. The `quality_bitmask` parameter is ignored but accepted for
    compatibility with other data format readers.

    More information: https://archive.stsci.edu/hlsp/cdips

    Parameters
    ----------
    filename : str
        Local path or remote url of CDIPS light curve FITS file.
    flux_column : str
        'IFL#', 'IRM#', 'TFA#', or 'PCA#' (# = 1, 2, or 3)
        Which column in the FITS file contains the preferred flux data?
    include_inst_errs: bool
        Whether to include the instrumental flux/magnitude errors
        (Errors are not provided for trend-filtered magnitudes)
    """
    ap = flux_column[-1]

    # Only the instrumental magnitudes are provided, and are not provided for
    # trend-filtered light curves. User should select whether to include the
    # instrumental errors or ignore them
    if include_inst_errs:
        # If fluxes are requested, return flux errors
        if flux_column[:-1].lower()=="ifl":
            flux_err_column = f"ife{ap}"
        # Otherwise magnitudes are being requested, return magnitude errors
        else:
            flux_err_column = f"ire{ap}"
    else:
        flux_err_column = ""

    # Set the appropriate error column for this aperture
    quality_column = f"irq{ap}"

    lc = read_generic_lightcurve(filename,
                                 time_column="tmid_bjd",
                                 flux_column=flux_column.lower(),
                                 flux_err_column=flux_err_column,
                                 quality_column=quality_column,
                                 time_format='btjd')

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119

    # CDIPS uses their own quality keywords instead of the default bitflags
    # Based on Bouma+2019, they filter out coarse point (4) and desat (32)
    # as well as other cadences flagged for particular sectors
    quality_mask = (lc['quality']=="G") | (lc['quality']=="0")
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "CDIPS"
    lc.meta['TARGETID'] = lc.meta.get('TICID')
    lc.meta['QUALITY_BITMASK'] = 36
    lc.meta['QUALITY_MASK'] = quality_mask

    return TessLightCurve(data=lc)
