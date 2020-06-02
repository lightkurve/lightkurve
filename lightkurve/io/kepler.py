import logging
import warnings

import numpy as np

from astropy.io import registry, fits
from astropy.table import Table
from astropy.time import Time

from ..lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from ..utils import KeplerQualityFlags, TessQualityFlags
from ..time import TimeBKJD, TimeBTJD


__all__ = ["read_kepler_lightcurve", "read_tess_lightcurve"]


log = logging.getLogger(__name__)


def _read_lightcurve_fits_file(filename, flux_column, flux_err_column,
                               quality_column='quality',
                               centroid_col_column='mom_centr1',
                               centroid_row_column='mom_centr2',
                               time_format=None):
    """Generic helper function to convert a Kepler ot TESS light curve file
    into a generic `LightCurve` object.
    """
    hdulist = fits.open(filename)
    hdu = hdulist[1]
    tab = Table.read(hdu, format='fits')

    # Some KEPLER files used to have a T column instead of TIME.
    if "T" in tab.colnames:
        tab.rename_column("T", "TIME")

    for colname in tab.colnames:
        # Ensure units have the correct astropy format
        if tab[colname].unit == 'e-/s':
            tab[colname].unit = 'electron/s'
        if tab[colname].unit == 'pixels':
            tab[colname].unit = 'pixel'

        # Rename columns to lowercase
        tab.rename_column(colname, colname.lower())

    # We *have* to remove rows with TIME=NaN because the Astropy Time
    # object does not support the presence of NaNs.
    # Fortunately, such rows are always bad data.
    nans = np.isnan(tab['time'].data)
    if np.any(nans):
        log.debug('Ignoring {} rows with NaN times'.format(np.sum(nans)))
    tab = tab[~nans]

    # Prepare a special time column
    if not time_format:
        if hdu.header.get('BJDREFI') == 2454833:
            time_format = 'bkjd'
        elif hdu.header.get('BJDREFI') == 2457000:
            time_format = 'btjd'
        else:
            raise ValueError(f"Input file has unclear time format: {filename}")
    time = Time(tab['time'].data,
                scale=hdu.header.get('TIMESYS', 'tdb').lower(),
                format=time_format)
    tab.remove_column('time')

    # For backwards compatibility with Lightkurve v1.x,
    # we make sure standard columns and attributes exist.
    if 'flux' not in tab.columns:
        tab.add_column(tab[flux_column], name="flux", index=0)
    if 'flux_err' not in tab.columns:
        tab.add_column(tab[flux_err_column], name="flux_err", index=1)
    if 'quality' not in tab.columns and quality_column in tab.columns:
        tab.add_column(tab[quality_column], name="quality", index=2)
    if 'centroid_col' not in tab.columns and centroid_col_column in tab.columns:
        tab.add_column(tab[centroid_col_column], name="centroid_col", index=3)
    if 'centroid_row' not in tab.columns and centroid_row_column in tab.columns:
        tab.add_column(tab[centroid_row_column], name="centroid_row", index=4)

    tab.meta['label'] = hdulist[0].header['OBJECT']
    tab.meta['mission'] = hdulist[0].header['TELESCOP']
    tab.meta['ra'] = hdulist[0].header['RA_OBJ']
    tab.meta['dec'] = hdulist[0].header['DEC_OBJ']
    tab.meta['filename'] = filename

    return LightCurve(time=time, data=tab)


def read_kepler_lightcurve(filename,
                           flux_column="pdcsap_flux",
                           flux_err_column="pdcsap_flux_err",
                           quality_bitmask="default"):
    """Returns a KeplerLightCurve.

    Parameters
    ----------
    filename : str
        Local path or remote url of a Kepler light curve FITS file.
    flux_column : 'pdcsap_flux' or 'sap_flux'
        Which column in the FITS file contains the preferred flux data?
    flux_err_column : 'pdcsap_flux_err' or 'sap_flux_err'
            Which column in the FITS file contains the preferred flux_err data?
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

        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    """
    lc = _read_lightcurve_fits_file(filename,
                                    flux_column=flux_column,
                                    flux_err_column=flux_err_column,
                                    quality_column='sap_quality',
                                    time_format='bkjd')

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=lc['sap_quality'],
                                bitmask=quality_bitmask)
    lc = lc[quality_mask]

    lc.meta['targetid'] = lc.meta['KEPLERID']
    lc.meta['quality_bitmask'] = quality_bitmask
    lc.meta['quality_mask'] = quality_mask
    return KeplerLightCurve(data=lc)


def read_tess_lightcurve(filename,
                         flux_column="pdcsap_flux",
                         flux_err_column="pdcsap_flux_err",
                         quality_bitmask="default"):
    """Returns a `TessLightCurve`.

    Parameters
    ----------
    filename : str
        Local path or remote url of a Kepler light curve FITS file.
    flux_column : 'pdcsap_flux' or 'sap_flux'
        Which column in the FITS file contains the preferred flux data?
    flux_err_column : 'pdcsap_flux_err' or 'sap_flux_err'
            Which column in the FITS file contains the preferred flux_err data?
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

        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    """
    lc = _read_lightcurve_fits_file(filename,
                                    flux_column=flux_column,
                                    flux_err_column=flux_err_column,
                                    time_format='btjd')

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = TessQualityFlags.create_quality_mask(
                                quality_array=lc['quality'],
                                bitmask=quality_bitmask)
    lc = lc[quality_mask]

    lc.meta['targetid'] = lc.meta['TICID']
    lc.meta['quality_bitmask'] = quality_bitmask
    lc.meta['quality_mask'] = quality_mask
    return TessLightCurve(data=lc)


"""ADD READERS TO THE REGISTRY"""
from astropy.io.registry import IORegistryError
try:
    registry.register_reader('kepler', LightCurve, read_kepler_lightcurve)
    registry.register_reader('tess', LightCurve, read_tess_lightcurve)
except IORegistryError:
    pass  # necessary to enable autoreload during debugging
