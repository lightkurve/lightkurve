"""Read a generic FITS table containing a light curve."""
import logging

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import numpy as np

from ..utils import validate_method
from ..lightcurve import LightCurve


log = logging.getLogger(__name__)


def read_generic_lightcurve(filename, flux_column,
                            quality_column='quality',
                            cadenceno_column='cadenceno',
                            centroid_col_column='mom_centr1',
                            centroid_row_column='mom_centr2',
                            time_format=None,
                            ext=1):
    """Generic helper function to convert a Kepler ot TESS light curve file
    into a generic `LightCurve` object.
    """
    if isinstance(filename, fits.HDUList):
        hdulist = filename  # Allow HDUList to be passed
    else:
        hdulist = fits.open(filename)

    # Raise an exception if the requested extension is invalid
    if isinstance(ext, str):
        validate_method(ext, supported_methods=[hdu.name.lower() for hdu in hdulist])
    hdu = hdulist[ext]
    tab = Table.read(hdu, format='fits')

    # Make sure the meta data also includes header fields from extension #0
    tab.meta.update(hdulist[0].header)

    # Use lowercase for meta data fields
    tab.meta = {k.lower(): v for k, v in tab.meta.items()}

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
    if 'flux_err' not in tab.columns and f"{flux_column}_err" in tab.columns:
        tab.add_column(tab[f"{flux_column}_err"], name="flux_err", index=1)
    if 'quality' not in tab.columns and quality_column in tab.columns:
        tab.add_column(tab[quality_column], name="quality", index=2)
    if 'cadenceno' not in tab.columns and cadenceno_column in tab.columns:
        tab.add_column(tab[cadenceno_column], name="cadenceno", index=3)
    if 'centroid_col' not in tab.columns and centroid_col_column in tab.columns:
        tab.add_column(tab[centroid_col_column], name="centroid_col", index=4)
    if 'centroid_row' not in tab.columns and centroid_row_column in tab.columns:
        tab.add_column(tab[centroid_row_column], name="centroid_row", index=5)

    tab.meta['label'] = hdulist[0].header.get('OBJECT')
    tab.meta['mission'] = hdulist[0].header.get('MISSION', hdulist[0].header.get('TELESCOP'))
    tab.meta['ra'] = hdulist[0].header.get('RA_OBJ')
    tab.meta['dec'] = hdulist[0].header.get('DEC_OBJ')
    tab.meta['filename'] = filename

    return LightCurve(time=time, data=tab)
