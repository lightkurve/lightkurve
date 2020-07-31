"""Reader for K2 EVEREST light curves."""
from .generic import read_generic_lightcurve
from .. import KeplerLightCurve


def read_everest_lightcurve(filename,
                            flux_column="flux",
                            quality_bitmask="default",
                            **kwargs):
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
    lc = read_generic_lightcurve(filename,
                                 flux_column=flux_column,
                                 quality_column='quality',
                                 cadenceno_column='cadn',
                                 time_format='bkjd')

    #tab.rename_column('flux', 'flux_original')
    #tab.add_column(tab["fraw_err"], name="flux_err", index=1)

    lc.meta['label'] = '{} (EVEREST)'.format(lc.meta.get("object"))
    lc.meta['targetid'] = lc.meta.get('keplerid')

    return KeplerLightCurve(data=lc, **kwargs)
