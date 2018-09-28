from __future__ import division, print_function
import os
import logging
import numpy as np

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from .lightcurve import KeplerLightCurve, TessLightCurve
from .lightcurvefile import KeplerLightCurveFile
from .targetpixelfile import TargetPixelFile, KeplerTargetPixelFile

from . import PACKAGEDIR
log = logging.getLogger(__name__)

class SearchResult(object):
    """

    """

    def __init__(self, info):

        self.info = info

    @property
    def ID(self):
        return np.asarray(np.unique(self.info['obsid']), dtype='int')

    @property
    def target_name(self):
        return np.asarray(np.unique(self.info['target_name']))

    @property
    def pos(self):
        coords = []
        for i,ra in enumerate(self.info['s_ra']):
            coords.append([ra, self.info['s_dec'][i]])
        if len(coords) == 1:
            return np.asarray(coords)[0]
        else:
            return np.asarray(coords)


    def download(self, type, quality_bitmask='default', **kwargs):
        """

        """
        if type == None:
            raise ValueError("Choose a filetype of 'Target Pixel' or 'Lightcurve'")
        elif type == "tpf":
            filetype = "Target Pixel"
        elif type == "lc":
            filetype = "Lightcurve"

        obsids = np.asarray(self.info['obsid'])
        products = Observations.get_product_list(self.info)
        order = [np.where(products['parent_obsid'] == o)[0] for o in obsids]
        order = [item for sublist in order for item in sublist]
        products = self._mask_products(products[order])

        path = Observations.download_products(products, mrp_only=False)['Local Path']

        if len(path) == 1:
            if type == "tpf":
                return KeplerTargetPixelFile(path[0],
                                             quality_bitmask=quality_bitmask,
                                             **kwargs)
            elif type == "lc":
                return KeplerLightCurveFile(path[0],
                                        quality_bitmask=quality_bitmask,
                                        **kwargs)

        if type == "tpf":
            return KeplerTargetPixelFile(path[0],
                                         quality_bitmask=quality_bitmask,
                                         **kwargs)
        elif type == "lc":
            return KeplerLightCurveFile(path[0],
                                    quality_bitmask=quality_bitmask,
                                    **kwargs)

    def _mask_products(self, products, filetype='Target Pixel', cadence='long', quarter=None,
                      campaign=None, searchtype='single', radius=1, targetlimit=1):
        """

        """
        # Value for the quarter or campaign
        qoc = campaign if campaign is not None else quarter

        # Ensure quarter or campaign is iterable.
        if (campaign is not None) | (quarter is not None):
            qoc = np.atleast_1d(np.asarray(qoc, dtype=int))

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

        return products


def _query_kepler_products(target, searchtype='single', radius=1):
    """
    Helper function for `search_kepler_products`.

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

    # If performing a cone search and multiple targets are desired, query_criteria()
    # must be called with `objectname` argument
    if searchtype == 'cone':
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
            target_obs = Observations.query_criteria(target_name=target_name,
                                                     radius='{} deg'.format(.0001),
                                                     project=["Kepler", "K2"],
                                                     obs_collection=["Kepler", "K2"])
            ra = target_obs['s_ra'][0]
            dec = target_obs['s_ra'][0]
            obs = Observations.query_criteria(coordinates='{} {}'.format(ra, dec),
                                              radius='{} deg'.format(radius/3600),
                                              project=["Kepler", "K2"],
                                              obs_collection=["Kepler", "K2"])
        except ValueError:
            # If `target` did not look like a KIC or EPIC ID, then we let MAST
            # resolve the target name to a sky position. Convert radius from arcsec
            # to degrees for query_criteria().
            try:
                obs = Observations.query_criteria(objectname=target,
                                                  radius='{} deg'.format(radius/3600),
                                                  project=["Kepler", "K2"],
                                                  obs_collection=["Kepler", "K2"])
                # Make sure the final table is in DISTANCE order
                obs.sort('distance')
            except ResolverError as exc:
                raise ArchiveError(exc)

    elif searchtype == 'single':
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
            # resolve the target name to a sky position. Convert radius from arcsec
            # to degrees for query_criteria().
            try:
                obs = Observations.query_criteria(objectname=target,
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

    return obs


def _search_kepler_products(target, filetype='Target Pixel', cadence='long', quarter=None,
                            campaign=None, searchtype='single', radius=1, targetlimit=1):
    """
    Returns a table of Kepler or K2 Target Pixel Files or Lightcurve Files
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

    suffix = []
    for f in ['Target Pixel', 'Lightcurve']:
        if cadence in ['short', 'sc']:
            suffix.append("{} Short".format(f))
        elif cadence in ['any', 'both']:
            suffix.append("{}".format(f))
        else:
            suffix.append("{} Long".format(f))

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
    for s in suffix:
        mask |= np.array([s in desc for desc in products['description']])
        mask &= np.array(['tar' not in desc for desc in products['description']])
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
        path = _query_kepler_products(
            target=target, searchtype='single', radius=1.)

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
        path = _query_kepler_products(
            target=target, searchtype='cone', radius=radius)

    return SearchResult(path)
