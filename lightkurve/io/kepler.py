import logging
import warnings

import numpy as np

from astropy.io import registry, fits
from astropy.table import Table
from astropy.time import Time

from ..lightcurve import LightCurve, KeplerLightCurve
from ..utils import KeplerQualityFlags
from ..time import TimeBKJD


__all__ = ["read_kepler_lightcurve"]


log = logging.getLogger(__name__)


def read_kepler_lightcurve(filename,
                           flux_column="pdcsap_flux",
                           flux_err_column="pdcsap_flux_err",
                           quality_bitmask="default"):
    """Returns a KeplerLightCurve

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
    hdulist = fits.open(filename)
    hdu = hdulist[1]
    tab = Table.read(hdu, format='fits')

    # Some KEPLER files have a T column instead of TIME.
    if "T" in tab.colnames:
        tab.rename_column("T", "TIME")

    for colname in tab.colnames:
        # Ensure units have the correct format
        if tab[colname].unit == 'e-/s':
            tab[colname].unit = 'electron/s'
        if tab[colname].unit == 'pixels':
            tab[colname].unit = 'pixel'

        # Rename columns to lowercase
        tab.rename_column(colname, colname.lower())

    # Filter out NaN rows
    nans = np.isnan(tab['time'].data)
    if np.any(nans):
        log.debug('Ignoring {} rows with NaN times'.format(np.sum(nans)))
    tab = tab[~nans]

    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=tab['sap_quality'],
                                bitmask=quality_bitmask)

    tab = tab[quality_mask]

    # Prepare a special time column, which will be passed to LightCurve at the end
    time = Time(tab['time'].data,
                scale=hdu.header['TIMESYS'].lower(),
                format='bkjd')
    tab.remove_column('time')

    # For backwards compatibility with Lightkurve v1.x,
    # we make sure standard columns and attributes exist.
    tab.add_column(tab[flux_column], name="flux", index=0)
    tab.add_column(tab[flux_err_column], name="flux_err", index=1)
    tab.add_column(tab['sap_quality'], name="quality", index=2)
    tab.add_column(tab['mom_centr1'], name="centroid_col", index=3)
    tab.add_column(tab['mom_centr2'], name="centroid_row", index=4)

    tab.meta['targetid'] = hdulist[0].header['KEPLERID']
    tab.meta['label'] = hdulist[0].header['OBJECT']
    tab.meta['mission'] = hdulist[0].header['TELESCOP']
    tab.meta['ra'] = hdulist[0].header['RA_OBJ']
    tab.meta['dec'] = hdulist[0].header['DEC_OBJ']
    tab.meta['quality_bitmask'] = quality_bitmask
    tab.meta['quality_mask'] = quality_mask
    tab.meta['filename'] = filename

    column_order = tab.columns.keys()
    tab = tab[column_order]

    return LightCurve(time=time, data=tab)


from astropy.io.registry import IORegistryError
try:
    registry.register_reader('kepler', LightCurve, read_kepler_lightcurve)
except IORegistryError:
    pass  # necessary to enable autoreload during debugging
