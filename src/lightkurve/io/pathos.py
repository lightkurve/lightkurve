"""Reader for A PSF-Based Approach to TESS High Quality Data Of Stellar Clusters (PATHOS) light curve files.

Website: https://archive.stsci.edu/hlsp/pathos
A product description file wasn't obvious on the MAST website
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_pathos_lightcurve(
    filename, flux_column="PSF_FLUX_COR", quality_bitmask="default"
):
    """Returns a `TessLightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of PATHOS light curve FITS file.
    flux_column : 'psf_flux_cor' or 'ap#_flux_cor' (# = 1, 2, 3, or 4)
        or 'psf_flux_raw' or 'ap#_flux_raw' (# = 1, 2, 3, or 4)
        Which column in the FITS file contains the preferred flux data?
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=1130799`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=1664431`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=2096639`). This mask is not recommended.

        See the :class:`TessQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(
        filename,
        flux_column=flux_column.lower(),
        time_format="btjd",
        quality_column="DQUALITY",
    )

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["dquality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["TARGETID"] = lc.meta.get("TICID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask

    # QLP light curves are normalized by default
    lc.meta["NORMALIZED"] = True

    return TessLightCurve(data=lc)
