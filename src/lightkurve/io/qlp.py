"""Reader for MIT Quicklook Pipeline (QLP) light curve files.

Website: http://archive.stsci.edu/hlsp/qlp
Product description: https://archive.stsci.edu/hlsps/qlp/hlsp_qlp_tess_ffi_all_tess_v1_data-prod-desc.pdf
"""
from ..lightcurve import TessLightCurve
from ..utils import TessQualityFlags

from .generic import read_generic_lightcurve


def read_qlp_lightcurve(filename, flux_column="sap_flux", flux_err_column=None, quality_bitmask="default"):
    """Returns a `~lightkurve.lightcurve.LightCurve` object given a light curve file from the MIT Quicklook Pipeline (QLP).

    By default, QLP's `sap_flux` column is used to populate the `flux` values,
    and `kspsap_flux_err` / `det_flux_err` is used to populate `flux_err`. For a discussion
    related to this choice, see https://github.com/lightkurve/lightkurve/issues/1083

    For detrended flux, the columns are named with `kspsap_` prefix in sectors 1-55,
    and `det_` prefix in sectors 56+. Column `sys_rm_flux` is available in sectors 56+.

    More information: https://archive.stsci.edu/hlsp/qlp

    Parameters
    ----------
    filename : str
        Local path or remote url of a QLP light curve FITS file.
    flux_column : 'sap_flux', 'kspsap_flux', 'kspsap_flux_sml', 'kspsap_flux_lag', 'det_flux', 'det_flux_sml', 'det_flux_lag', 'sys_rm_flux', or 'sap_bkg'
        Which column in the FITS file contains the preferred flux data?
        By default the "Simple Aperture Photometry" flux (sap_flux) is used.
    flux_err_column: 'kspsap_flux_err','det_flux_err', or 'sap_bkg_err'
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

        QLP-specific "Low precision points" (bit 13 in sectors 1-55, bit 31 in sectors 56+)
        is included in "hard" and "hardest" bitmasks.

        See the `~lightkurve.utils.TessQualityFlags` class for details on the bitmasks.
    """
    lc = read_generic_lightcurve(filename, flux_column=flux_column, flux_err_column=flux_err_column, time_format="btjd")
    if flux_err_column is None:
        if lc.meta.get("SECTOR", 0) >= 56:
            lc["flux_err"] = lc["det_flux_err"]
        else:
            lc["flux_err"] = lc["kspsap_flux_err"]

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
        quality_array=lc["quality"], bitmask=quality_bitmask
    )
    # QLP-specific quality_bitmask handling
    if quality_bitmask in ["hardest", "hard"]:
        if lc.meta.get("SECTOR", 0) >= 56:
            qlp_low_precision_bitmask = 2 ** 30
        else:
            # https://archive.stsci.edu/hlsps/qlp/hlsp_qlp_tess_ffi_all_tess_v1_data-prod-desc.pdf
            qlp_low_precision_bitmask = 2 ** 12
        q_mask2 = TessQualityFlags.create_quality_mask(
            quality_array=lc["quality"], bitmask=qlp_low_precision_bitmask)
        quality_mask = quality_mask & q_mask2
    lc = lc[quality_mask]

    lc.meta["AUTHOR"] = "QLP"
    lc.meta["TARGETID"] = lc.meta.get("TICID")
    lc.meta["QUALITY_BITMASK"] = quality_bitmask
    lc.meta["QUALITY_MASK"] = quality_mask

    # QLP light curves are normalized by default
    lc.meta["NORMALIZED"] = True

    return TessLightCurve(data=lc)
