"""Reader for MIT Quicklook Pipeline (QLP) light curve files.

Website: http://archive.stsci.edu/hlsp/qlp
Product description: https://archive.stsci.edu/hlsps/qlp/hlsp_qlp_tess_ffi_all_tess_v1_data-prod-desc.pdf
"""
from astropy.io.fits import HDUList

from ..lightcurve import LightCurve, TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve
from .spec import ReaderSpec

AUTHOR_NAME = "QLP"


def read_qlp_lightcurve(filename, flux_column="sap_flux", flux_err_column="kspsap_flux_err", quality_bitmask="default"):
    """Returns a `~lightkurve.lightcurve.LightCurve` object given a light curve file from the MIT Quicklook Pipeline (QLP).

    By default, QLP's `sap_flux` column is used to populate the `flux` values,
    and 'kspsap_flux_err' is used to populate `flux_err`. For a discussion
    related to this choice, see https://github.com/lightkurve/lightkurve/issues/1083

    More information: https://archive.stsci.edu/hlsp/qlp

    Parameters
    ----------
    filename : str
        Local path or remote url of a QLP light curve FITS file.
    flux_column : 'sap_flux', 'kspsap_flux', 'kspsap_flux_sml', 'kspsap_flux_lag', or 'sap_bkg'
        Which column in the FITS file contains the preferred flux data?
        By default the "Simple Aperture Photometry" flux (sap_flux) is used.
    flux_err_column: 'kspsap_flux_err', or 'sap_bkg_err'
      Which column in the FITS file contains the preferred flux_err data?
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored.
            * "default": cadences with severe quality issues will be ignored.
            * "hard": more conservative choice of flags to ignore.
              This is known to remove good data.
            * "hardest": removes all data that has been flagged.
              This mask is not recommended.

        See the `~lightkurve.utils.TessQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(filename, flux_column=flux_column, flux_err_column=flux_err_column, time_format="btjd")

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = AUTHOR_NAME
    lc.meta["TARGETID"] = lc.meta.get("TICID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask

    # QLP light curves are normalized by default
    lc.meta["NORMALIZED"] = True

    return TessLightCurve(data=lc)


#
# Hooks to add the reader
#

FORMAT_NAME = "qlp"

READER_SPEC = ReaderSpec(FORMAT_NAME, LightCurve, read_qlp_lightcurve)


def detect_filetype(hdulist: HDUList) -> str:
    if "mit/qlp" in hdulist[0].header.get("origin", "").lower():
        return FORMAT_NAME  # refers to the format name in the reader spec above


# Which external links should we display in the SearchResult repr?
AUTHOR_LINKS = {
  AUTHOR_NAME: "https://archive.stsci.edu/hlsp/qlp",
}

