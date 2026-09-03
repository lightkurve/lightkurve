"""Reader for TESS All-Sky Rotation Survey (TARS) light curve files.

Website: https://archive.stsci.edu/hlsp/tars
"""
import numpy as np

from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_tars_lightcurve(filename, flux_column="flux", quality_bitmask="default"):
    """Returns a `~lightkurve.lightcurve.LightCurve` object given a light curve file
    from the TESS All-Sky Rotation Survey (TARS) HLSP.

    TARS light curve files contain normalized flux values extracted from
    TESS Full Frame Images. The files provide only ``TIME`` and ``FLUX``
    columns; no flux errors or quality flags are included. Because no
    per-cadence quality flags are available, the returned light curve is
    assigned a ``quality`` column filled with zeros, and the
    ``quality_bitmask`` parameter has no effect on which cadences are
    returned.

    More information: https://archive.stsci.edu/hlsp/tars

    Parameters
    ----------
    filename : str
        Local path or remote url of a TARS light curve FITS file.
    flux_column : str
        Which column in the FITS file contains the preferred flux data?
        By default the normalized flux (``flux``) is used; it is the only
        flux column provided by this HLSP.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. Accepted for consistency with other
        readers, but TARS files do not provide quality flags, so all cadences
        are always returned.
    """
    lc = read_generic_lightcurve(filename, flux_column=flux_column, time_format="btjd")

    # TARS files do not provide per-cadence quality flags,
    # so we assign a zero (i.e. "good") quality value to every cadence.
    lc["quality"] = np.zeros(len(lc), dtype=np.int32)

    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "TARS"
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask

    # TARS light curves are normalized by default
    lc.meta["NORMALIZED"] = True

    tic = lc.meta.get("TICID")
    if tic is not None:
        tic = int(tic)
        # compatibility with SPOC, QLP, etc.
        lc.meta["TARGETID"] = tic
        lc.meta["TICID"] = tic
        lc.meta["OBJECT"] = f"TIC {tic}"
        # for Lightkurve's plotting methods
        lc.meta["LABEL"] = f"TIC {tic}"

    return TessLightCurve(data=lc)
