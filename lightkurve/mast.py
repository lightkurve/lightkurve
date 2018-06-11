"""Functions which wrap `astroquery.mast` to obtain Kepler/K2 data from MAST."""

from __future__ import division, print_function
import os
import logging
import numpy as np

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from . import PACKAGEDIR
log = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Raised if there is a problem accessing data."""
    pass


def download_products(products):
    """Download data products from MAST.

    Downloads will be cached using astroquery.mast's caching system.

    Parameters
    ----------
    products : astropy.Table
        A table detailing the products to be downloaded, i.e. the output
        of `search_products` or `search_kepler_tpf_products`.

    Returns
    -------
    paths : astropy.Table.Column
        Detailing the paths of the downloaded or cached products.
    """
    # Note: by default, MAST will only let us download "Minimum Recommended
    # Products" (MRPs), which do not include e.g. Target Pixel Files.
    # We need to set `mrp=False` to ensure MAST downloads whatever we want.
    dl = Observations.download_products(products, mrp_only=False)
    return dl['Local Path']


def _query_kepler_products(target, radius=1):
    """Helper function for `search_kepler_products`.

    Returns a table of Kepler/K2 pipeline products for a given target.

    Raises an ArchiveError if no products are found.

    Parameters
    ----------
    target : str or int
        If the value is an integer in a specific range, we'll assume it is
        a KIC or EPIC ID.  Otherwise we will pass it to MAST as an objectname.

    Returns
    -------
    products : astropy.Table
        Table detailing the available data products.
    """
    # If passed a SkyCoord, convert it to an RA and Dec
    if isinstance(target, SkyCoord):
        target = '{}, {}'.format(target.ra.deg, target.dec.deg)

    try:
        # If `target` looks like a KIC or EPIC ID, we will pass the exact
        # `target_name` under which MAST will know the object.
        target = int(target)
        if (target > 0) and (target < 200000000):
            target_name = 'kplr{:09d}'.format(target)
        elif (target > 200000000) and (target < 300000000):
            target_name = 'ktwo{:09d}'.format(target)
        else:
            raise ValueError("{:09d}: not in the KIC or EPIC ID range".format(target))
        obs = Observations.query_criteria(target_name=target_name,
                                          project=["Kepler", "K2"],
                                          obs_collection=["Kepler", "K2"])
    except ValueError:
        # If `target` did not look like a KIC or EPIC ID, then we let MAST
        # resolve the target name to a sky position.
        try:
            obs = Observations.query_criteria(objectname=target,
                                              radius='{} arcsec'.format(radius),
                                              project=["Kepler", "K2"],
                                              obs_collection=["Kepler", "K2"])
            # Make sure the final table is in DISTANCE order
            obs.sort('distance')
        except ResolverError as exc:
            raise ArchiveError(exc)

    obsids = np.asarray(obs['obsid'])
    products = Observations.get_product_list(obs)
    order = [np.where(products['parent_obsid'] == o)[0] for o in obsids]
    order = [item for sublist in order for item in sublist]
    products = products[order]
    return products


def search_kepler_products(target, filetype='Target Pixel', cadence='long', quarter=None,
                           campaign=None, radius=1, targetlimit=1):
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

    products = _query_kepler_products(target, radius)
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
    if targetlimit is not None:
        ids = np.asarray([p.split('/')[-1].split('-')[0].split('_')[0][4:]
                          for p in products['dataURI']], dtype=int)
        if len(np.unique(ids)) < targetlimit:
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


def download_kepler_products(target, filetype='Target Pixel', cadence='long',
                             quarter=None, month=None, campaign=None, radius=1.,
                             targetlimit=1, **kwargs):
    """Download and cache files from from the Kepler/K2 data archive at MAST.

    Returns paths to the cached files.

    Raises an `ArchiveError` if a unique TPF cannot be found.  For example,
    this is the case if a target was observed in multiple Quarters and the
    quarter parameter is unspecified.

    Parameters
    ----------
    target : str or int
        KIC/EPIC ID or object name.

    filetype : 'Target Pixel' or 'Lightcurve'
        Whether to return TPFs of LCs
    cadence : str
        'long' or 'short'.
    quarter, campaign : int, list of int or 'all'
        Kepler Quarter or K2 Campaign number.
    month : 1, 2, 3, list of int or 'all'
        For Kepler's prime mission, there are three short-cadence
        Target Pixel Files for each quarter, each covering one month.
        Hence, if cadence='short' you need to specify month=1, 2, or 3.
    radius : float
        Search radius in arcseconds. Default is 1 arcsecond.
    targetlimit : None or int
        Limit the number of returned target pixel files. If none, no limit
        is set
    kwargs : dict
        Keywords arguments passed to `KeplerTargetPixelFile`.

    Returns
    -------
    path : str or list of strs
    """
    # Make sure astroquery uses the same level of verbosity
    logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

    # If we are asking for all cadences (long and short) and didn't specify a month, override it.
    if (cadence in ['any', 'both']) & (month is None):
        month = np.arange(3) + 1

    # anys and all should be a list of numbers
    if month in ['all', 'any']:
        month = np.arange(3) + 1
    if campaign in ['all', 'any']:
        campaign = np.arange(0, 22)
    if quarter in ['all', 'any']:
        quarter = np.arange(0, 18)
    if month is not None:
        if not hasattr(month, '__iter__'):
            month = [month]

    # Grab the products
    products = search_kepler_products(target=target, filetype=filetype, cadence=cadence,
                                      quarter=quarter, campaign=campaign,
                                      radius=radius, targetlimit=targetlimit)
    if len(products) == 0:
        raise ArchiveError("No {} File found for {} at MAST.".format(filetype, target))
    products.sort(['order', 'dates', 'qoc'])

    # For Kepler short cadence data there are additional rules, so find anywhere
    # where there is short cadence data...
    scmask = np.asarray(['Short' in d for d in products['description']]) &\
        np.asarray(['kplr' in d for d in products['dataURI']])
    if np.any(scmask):
        # Error check the user if there's multiple months and they didn't ask
        # for a specific one
        if month is None:
            raise ArchiveError("Found {} different Target Pixel Files "
                               "for target {} in Quarter {}. "
                               "Please specify the month (1, 2, or 3)."
                               "".format(len(products), target, quarter))
        # Get the short cadence date lookup table.
        table = ascii.read(os.path.join(PACKAGEDIR, 'data', 'short_cadence_month_lookup.csv'))
        # Grab the dates of each of the short cadence files. Make sure every entry
        # has the correct month
        finalmask = np.ones(len(products), dtype=bool)
        for c in np.unique(products[scmask]['qoc']):
            ok = (products['qoc'] == c) & (scmask)
            mask = np.zeros(np.shape(products[ok])[0], dtype=bool)
            for m in month:
                udate = (table['StartTime'][np.where(
                    (table['Month'] == m) & (table['Quarter'] == c))[0][0]])
                mask |= np.asarray(products['dates'][ok]) == udate
            finalmask[ok] = mask
        products = products[finalmask]
        # Sort by id, then date and quarter
        products.sort(['order', 'dates', 'qoc'])
        if len(products) < 1:
            raise ArchiveError("No {} File found for {} "
                               "at month {} at MAST.".format(filetype, target, month))

    # If there is no specified quarter but there are many campaigns/quarters
    # returned, warn the user with an error
    if (len(np.unique(products['qoc'])) > 1) & (campaign is None) & (quarter is None):
        raise ArchiveError("Found {} different Target Pixel Files "
                           "for target {}. Please specify quarter/month "
                           "or campaign number."
                           "".format(len(products), target))

    # Otherwise download all the files
    log.debug('Found {} File(s)'.format(np.shape(products)[0]))
    path = download_products(products)
    return list(path)
