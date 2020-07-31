"""Reader for K2 EVEREST light curves."""
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table

from .. import KeplerLightCurve


def read_everest_lightcurve(filename, **kwargs):
    """Read an EVEREST light curve file.

    More information: https://archive.stsci.edu/hlsp/everest

    Parameters
    ----------
    filename : str
        Path or URL of a K2SFF light curve FITS file.

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    if isinstance(filename, fits.HDUList):
        hdulist = filename  # Allow HDUList to be passed
    else:
        hdulist = fits.open(filename)

    hdu = hdulist[1]
    tab = Table.read(hdu, format='fits')

    # Make sure the meta data also includes header fields from extension #0
    tab.meta.update(hdulist[0].header)

    # Use lowercase for meta data fields
    tab.meta = {k.lower(): v for k, v in tab.meta.items()}

    # Rename columns to lowercase
    for colname in tab.colnames:
        tab.rename_column(colname, colname.lower())

    tab.rename_column('flux', 'flux_original')

    tab.add_column(tab["fcor"], name="flux", index=0)
    tab.add_column(tab["fraw_err"], name="flux_err", index=1)

    tab['flux'].unit = ""  # EVEREST light curves are normalized
    tab.remove_column('fcor')
    tab.remove_column('fraw_err')

    # Prepare a special time column
    time = Time(tab['time'].data,
                scale=hdu.header.get('TIMESYS', 'tdb').lower(),
                format='bkjd')
    tab.remove_column('time')

    tab.rename_column("cadn", "cadenceno")

    tab.meta['label'] = '{} (EVEREST)'.format(tab.meta.get("object"))
    tab.meta['targetid'] = tab.meta.get('keplerid')
    tab.meta['ra'] = tab.meta.get('ra_obj')
    tab.meta['dec'] = tab.meta.get('dec_obj')
    tab.meta['filename'] = filename

    """
    # Filter out poor-quality data
    # NOTE: Unfortunately Astropy Table masking does not yet work for columns
    # that are Quantity objects, so for now we remove poor-quality data instead
    # of masking. Details: https://github.com/astropy/astropy/issues/10119
    quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=lc['sap_quality'],
                                bitmask=quality_bitmask)
    """

    return KeplerLightCurve(time=time, data=tab, **kwargs)
