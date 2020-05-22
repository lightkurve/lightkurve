import warnings

import numpy as np

from astropy.io import registry, fits
from astropy.table import Table
from astropy.time import Time

from ..lightcurve import LightCurve
from ..time import TimeBKJD


__all__ = ["read_kepler_lightcurve"]


def read_kepler_lightcurve(filename, flux_column="pdcsap_flux",
                           flux_err_column="pdcsap_flux_err"):
    """Returns a KeplerLightCurve
    """
    hdulist = fits.open(filename)
    hdu = hdulist[1]
    tab = Table.read(hdu, format='fits')

    # Some KEPLER files have a T column instead of TIME.
    if "T" in tab.colnames:
        tab.rename_column("T", "TIME")

    for colname in tab.colnames:
        # Fix units
        if tab[colname].unit == 'e-/s':
            tab[colname].unit = 'electron/s'
        if tab[colname].unit == 'pixels':
            tab[colname].unit = 'pixel'

        # Rename columns to lowercase
        tab.rename_column(colname, colname.lower())

    # Filter out NaN rows
    nans = np.isnan(tab['time'].data)
    if np.any(nans):
        warnings.warn('Ignoring {} rows with NaN times'.format(np.sum(nans)))
    tab = tab[~nans]

    time = Time(tab['time'].data,
                scale=hdu.header['TIMESYS'].lower(),
                format='bkjd')

    # Remove original time column
    tab.remove_column('time')
    tab.meta['targetid'] = hdulist[0].header['KEPLERID']
    tab.meta['label'] = hdulist[0].header['OBJECT']
    tab.meta['ra'] = hdulist[0].header['RA_OBJ']
    tab.meta['dec'] = hdulist[0].header['DEC_OBJ']

    return LightCurve(time=time,
                      flux=tab[flux_column],
                      flux_err=tab[flux_err_column],
                      data=tab)


from astropy.io.registry import IORegistryError
try:
    registry.register_reader('kepler', LightCurve, read_kepler_lightcurve)
except IORegistryError:
    pass  # necessary to enable autoreload during debugging
