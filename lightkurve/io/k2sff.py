"""Reader function for K2SFF community light curve products."""
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import numpy as np

from .. import KeplerLightCurve
from ..utils import validate_method


def read_k2sff_lightcurve(filename, ext="BESTAPER", **kwargs):
    """Read a K2SFF light curve file.

    More information: https://archive.stsci.edu/hlsp/k2sff

    Parameters
    ----------
    filename : str
        Path or URL of a K2SFF light curve FITS file.
    ext : str
        Version of the light curve to use.  Valid options include "BESTAPER",
        "CIRC_APER0" through "CIRC_APER9", and "PRF_APER0" through "PRF_APER9".

    Returns
    -------
    lc : `KeplerLightCurve`
        A populated light curve object.
    """
    if isinstance(filename, fits.HDUList):
        hdulist = filename  # Allow HDUList to be passed
    else:
        hdulist = fits.open(filename)

    # Raise an exception if the requested extension is invalid
    validate_method(ext, supported_methods=[hdu.name.lower() for hdu in hdulist])

    hdu = hdulist[ext]
    tab = Table.read(hdu, format='fits')

    # Make sure the meta data also includes header fields from extension #0
    tab.meta.update(hdulist[0].header)

    # Use lowercase for meta data fields
    tab.meta = {k.lower(): v for k, v in tab.meta.items()}

    # Rename columns to lowercase
    for colname in tab.colnames:
        tab.rename_column(colname, colname.lower())

    tab.add_column(tab["fcor"], name="flux", index=0)
    tab.add_column(np.nan, name="flux_err", index=1)

    tab['flux'].unit = ""  # SFF light curves are normalized
    tab.remove_column('fcor')

    # Prepare a special time column
    time = Time(tab['t'].data,
                scale=hdu.header.get('TIMESYS', 'tdb').lower(),
                format='bkjd')
    tab.remove_column('t')

    tab.meta['label'] = '{} (K2SFF)'.format(tab.meta.get("object"))
    tab.meta['targetid'] = tab.meta.get('keplerid')
    tab.meta['ra'] = tab.meta.get('ra_obj')
    tab.meta['dec'] = tab.meta.get('dec_obj')
    tab.meta['filename'] = filename

    return KeplerLightCurve(time=time, data=tab, **kwargs)
