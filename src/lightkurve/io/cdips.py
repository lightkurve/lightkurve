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
                            flux_column="IFL1",
                            include_inst_errs=True,
                            quality_bitmask=None):
    """Returns a TESS CDIPS `~lightkurve.lightcurve.LightCurve`.

    Note: CDIPS light curves have already had quality filtering applied, and
    do not provide the bitflags necessary for a user to apply a new bitmask.
    Therefore, frames corresponding to "momentum dumps and coarse point modes"
    are removed according to Bouma et al. 2019, and no other quality filtering
    is allowed. The `quality_bitmask` parameter is ignored but accepted for
    compatibility with other data format readers.

    There are several kinds of flux and  magnitudes provided. For consistancy
    we have chosen to display as default 'IFL1', the flux in aperture 1. 
    This is given in ADU. 
    The flux_err is also provided as 'IFE1' in ADU.

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

    # A user can chose to dipsplay the magnitudes or any of the other flux values,
    # They can do this by using the flux_column key
    # By default the flux_err for the given flux specified is returned
    
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

    #The time displayed is in BJD not BTJD. We can fix this by subtracting
    # 2457000
    lc["time"] = lc["time"]-2457000


    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119

    # CDIPS uses their own quality keywords instead of the default bitflags
    # Based on Bouma+2019, they filter out coarse point (4) and desat (32)
    # as well as other cadences flagged for particular sectors
    quality_mask = (lc['quality']=="G") | (lc['quality']=="0")
    lc = lc[quality_mask]


    #This makes sure that the TIC ID is obtained and returned as the plot label
    tic = lc.meta.get('TICID')
    if tic is not None:
        # compatibility with SPOC, QLP, etc.
        lc.meta["TARGETID"] = tic
        lc.meta["TICID"] = tic
        lc.meta["OBJECT"] = f"TIC {tic}"
        # for Lightkurve's plotting methods
        lc.meta["LABEL"] = f"TIC {tic}"

    
    lc.meta["AUTHOR"] = "CDIPS"
    lc.meta['QUALITY_BITMASK'] = 36
    lc.meta['QUALITY_MASK'] = quality_mask

    return TessLightCurve(data=lc)
