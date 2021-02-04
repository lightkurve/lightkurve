"""Reader for A PSF-Based Approach to TESS High Quality Data Of Stellar Clusters (PATHOS) light curve files.

Website: https://archive.stsci.edu/hlsp/pathos
A product description file wasn't obvious on the MAST website
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_tasoc_lightcurve(filename,
                            flux_column="FLUX_RAW",
                            quality_bitmask="default"):
    """Returns a `TessLightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of TASOC light curve FITS file.
    flux_column : 'flux_RAW' - this contains the T'DA extracted lightcurve,
    with no corrections applied to the raw light curves. Corrected lightcurves 
    may be a thing in the future as there is a flux_corr column.
    quality_bitmask : For now this always none - as no calibration applied

    """
    lc = read_generic_lightcurve(filename,
                                 flux_column=flux_column.lower(),
                                 time_format='btjd',
                                 quality_column="QUALITY")

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    #quality_mask = TessQualityFlags.create_quality_mask(
    #                            quality_array=lc['dquality'],
    #                            bitmask=quality_bitmask)
    #lc = lc[quality_mask]

    lc.meta['TARGETID'] = lc.meta.get('TICID')
    lc.meta['QUALITY_BITMASK'] = quality_bitmask
    #lc.meta['QUALITY_MASK'] = quality_mask

    # QLP light curves are normalized by default
    lc.meta['NORMALIZED'] = True

    return TessLightCurve(data=lc)
