from __future__ import division, print_function
import os
import logging
import numpy as np

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from .mast import download_kepler_products, _query_kepler_products

from . import PACKAGEDIR
log = logging.getLogger(__name__)

class SearchResult(object):

    def __init__(self, info):

        self.info = info

    @property
    def ID(self):
        return np.asarray(star.info['obsID'], dtype='int')

    def download(self, filetype=None):
        """

        """
        raise NotImplementedError("Functionality not yet added.")
        if filetype == "TargetPixelFile":
            pass
        elif filetype == "LightCurve":
            pass
        elif filetype == "LightCurveFile":
            pass


def _search_kepler_products(target, filetype='Target Pixel', cadence='long', quarter=None,
                            campaign=None, searchtype='single', radius=1, targetlimit=1):
    """Returns a table of Kepler or K2 Target Pixel Files or Lightcurve Files
     for a given target.

    Parameters
    ----------
    filetype : 'Target Pixel' or 'Lightcurve'
        Whether to return TPFs of LCs
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.
    quarter, campaign : int
        Specify the Kepler Quarter or K2 Campaign Number.
        If None, then return the products for all Quarters/Campaigns.
    radius : float
        Search radius in arcseconds
    targetlimit : None or int
        If multiple targets are present within `radius`, limit the number
        of returned TargetPixelFile objects to `targetlimit`.
        If `None`, no limit is applied.

    Returns
    -------
    products : astropy.Table
        Table detailing the available Target Pixel File products.
    """
    if filetype not in ['Target Pixel', 'Lightcurve']:
        raise ValueError("Choose a filetype of 'Target Pixel' or 'Lightcurve'")

    # Value for the quarter or campaign
    qoc = campaign if campaign is not None else quarter

    # Ensure quarter or campaign is iterable.
    if (campaign is not None) | (quarter is not None):
        qoc = np.atleast_1d(np.asarray(qoc, dtype=int))

    products = _query_kepler_products(target, searchtype=searchtype, radius=radius)
    # Because MAST doesn't let us query based on Kepler-specific meta data
    # fields, we need to identify short/long-cadence TPFs by their filename.
    if cadence in ['short', 'sc']:
        suffix = "{} Short".format(filetype)
    elif cadence in ['any', 'both']:
        suffix = "{}".format(filetype)
    else:
        suffix = "{} Long".format(filetype)

    # If there is nothing in the table, quit now.
    if len(products) == 0:
        return products


    # Identify the campaign or quarter by the description.
    if qoc is not None:
        mask = np.zeros(np.shape(products)[0], dtype=bool)
        for q in qoc:
            mask |= np.array([desc.lower().replace('-', '').endswith('q{}'.format(q)) or
                              'c{:02d}'.format(q) in desc.lower().replace('-', '') or
                              'c{:03d}'.format(q) in desc.lower().replace('-', '')
                              for desc in products['description']])
    else:
        mask = np.ones(np.shape(products)[0], dtype=bool)

    # Allow only the correct fits or fits.gz type
    mask &= np.array([desc.lower().endswith('fits') or
                      desc.lower().endswith('fits.gz')
                      for desc in products['dataURI']])

    # Allow only the correct cadence type
    mask &= np.array([suffix in desc for desc in products['description']])
    products = products[mask]

    # Add the quarter or campaign numbers
    qoc = np.asarray([p.split(' - ')[-1][1:].replace('-', '')
                      for p in products['description']], dtype=int)
    products['qoc'] = qoc
    # Add the dates of each short cadence observation to the product table.
    # Note this will not produce a date for ktwo observations, but will not break.
    dates = [p.split('/')[-1].split('-')[1].split('_')[0]
             for p in products['dataURI']]
    for idx, d in enumerate(dates):
        try:
            dates[idx] = float(d)
        except:
            dates[idx] = 0
    products['dates'] = np.asarray(dates)

    # Limit to the correct number of hits based on ID. If there are multiple versions
    # of the same ID, this shouldn't count towards the limit.
    # if targetlimit is not None:
    ids = np.asarray([p.split('/')[-1].split('-')[0].split('_')[0][4:]
                      for p in products['dataURI']], dtype=int)
    if targetlimit is None:
        pass
    elif len(np.unique(ids)) < targetlimit:
        log.warning('Target return limit set to {} '
                    'but only {} unique targets found. '
                    'Try increasing the search radius. '
                    '(Radius currently set to {} arcseconds)'
                    ''.format(targetlimit, len(np.unique(ids)), radius))
    okids = ids[np.sort(np.unique(ids, return_index=True)[1])[0:targetlimit]]
    mask = np.zeros(len(ids), dtype=bool)

    # Mask data.
    # Make sure they still appear in the same order.
    order = np.zeros(len(ids))
    for idx, okid in enumerate(okids):
        pos = ids == okid
        order[pos] = int(idx)
        mask |= pos
    products['order'] = order
    products = products[mask]

    return products

def search_target(target, cadence='long', quarter=None, month=None,
                 campaign=None, quality_bitmask='default', **kwargs):
    """
    Fetch a data table for a given Kepler target. `search_result` is
    intended to only return information for a single star. To perform
    a cone search of a region of sky, please use `search_region`.

    See the :class:`KeplerQualityFlags` class for details on the bitmasks.

    Parameters
    ----------
    target : str or int
        KIC/EPIC ID or object name.
    cadence : str
        'long' or 'short'.
    quarter, campaign : int, list of ints, or 'all'
        Kepler Quarter or K2 Campaign number.
    month : 1, 2, 3, list or 'all'
        For Kepler's prime mission, there are three short-cadence
        Target Pixel Files for each quarter, each covering one month.
        Hence, if cadence='short' you need to specify month=1, 2, or 3.
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
    kwargs : dict
        Keywords arguments passed to the constructor of
        :class:`KeplerTargetPixelFile`.

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    if os.path.exists(str(target)) or str(target).startswith('http'):
        log.warning('Warning: from_archive() is not intended to accept a '
                    'direct path, use KeplerTargetPixelFile(path) instead.')
        path = [target]
    else:
        path = _search_kepler_products(
            target=target, filetype='Target Pixel', cadence=cadence,
            quarter=quarter, campaign=campaign,
            searchtype='single', radius=1., targetlimit=1)

    return SearchResult(path)


def search_region(target, cadence='long', quarter=None, month=None,
                  campaign=None, radius=100., targetlimit=None,
                  quality_bitmask='default', **kwargs):
    """
    Fetch a data table for targets within a region of sky. Cone search is
    centered around the position of `target` and extends to a given `radius`.

    See the :class:`KeplerQualityFlags` class for details on the bitmasks.

    Parameters
    ----------
    target : str or int
        KIC/EPIC ID or object name.
    cadence : str
        'long' or 'short'.
    quarter, campaign : int, list of ints, or 'all'
        Kepler Quarter or K2 Campaign number.
    month : 1, 2, 3, list or 'all'
        For Kepler's prime mission, there are three short-cadence
        Target Pixel Files for each quarter, each covering one month.
        Hence, if cadence='short' you need to specify month=1, 2, or 3.
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
    kwargs : dict
        Keywords arguments passed to the constructor of
        :class:`KeplerTargetPixelFile`.

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """

    if os.path.exists(str(target)) or str(target).startswith('http'):
        log.warning('Warning: from_archive() is not intended to accept a '
                    'direct path, use KeplerTargetPixelFile(path) instead.')
        path = [target]
    else:
        path = _search_kepler_products(
            target=target, filetype='Target Pixel', cadence=cadence,
            quarter=quarter, campaign=campaign,
            searchtype='cone', radius=radius, targetlimit=targetlimit)

    return SearchResult(path)
