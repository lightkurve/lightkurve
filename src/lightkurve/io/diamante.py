"""DIAMANTE TESS HLSP light curve reader.

Documentation:
https://archive.stsci.edu/hlsp/diamante

Example:
lk.search_lightcurve("TOI 172")
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_diamante_lightcurve(filename, flux_column="LC1_AP1", quality_bitmask=None):
    """Returns a `TessLightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of DIAMANTE light curve FITS file.
    flux_column : str
        Column that will be used to populate the flux values.
        By default, "LC0_AP1" is used.
    """
    flux_column = flux_column.lower()
    lc = read_generic_lightcurve(
        filename, time_column="btjd",
        flux_column=flux_column,
        flux_err_column="e"+flux_column,
        quality_column="flag_ap1",
        time_format="btjd"
    )

    lc.meta["AUTHOR"] = "DIAMANTE"
    lc.meta["TARGETID"] = lc.meta.get("TICID")
    return TessLightCurve(data=lc)
