"""TESS Asteroseismic Science Operations Center - https://tasoc.dk
   TESS Data For Asteroseismology Lightcurves - https://archive.stsci.edu/hlsp/tasoc
   Data provided with this release have been extracted using the TASOC Photometry pipeline. The TASOC
   pipeline used to generate the data is open source and available on GitHub - https://github.com/tasoc
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_tasoc_lightcurve(filename, flux_column="FLUX_CORR", quality_bitmask=None):
    """Returns a TESS TASOC `~lightkurve.lightcurve.LightCurve`.

    More information: https://archive.stsci.edu/hlsp/tasoc

    Parameters
    ----------
    filename : str
        Local path or remote url of TASOC light curve FITS file.
    flux_column : str
        Column that will be used to populate the flux values.
        By default, "FLUX_CORR" is used. It contains the T'DA extracted lightcurve,
        with corrections applied to the raw light curves
    """
    lc = read_generic_lightcurve(
        filename, flux_column=flux_column.lower(), time_format="btjd"
    )

    lc.meta["AUTHOR"] = "TASOC"
    lc.meta["TARGETID"] = lc.meta.get("TICID")
    # TASOC light curves are normalized by default
    lc.meta["NORMALIZED"] = True
    return TessLightCurve(data=lc)
