from __future__ import division, print_function
import os
import logging
import numpy as np

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from astropy.table import unique, join
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from .lightcurvefile import KeplerLightCurveFile
from .targetpixelfile import KeplerTargetPixelFile
from .collections import TargetPixelFileCollection, LightCurveFileCollection

from . import PACKAGEDIR
log = logging.getLogger(__name__)


class SearchResult(object):
    """
    Defines a generic SearchResult class returned by `search_targetpixelfile` or `search_lightcurvefile`.

    Parameters
    ----------
    products : astropy table
        Astropy table returned by a join of the astroquery `Observations.query_criteria()`
        and `Observations.get_product_list()` methods.
    campaign : int or list
        Desired campaign of observation for data products
    quarter : int or list
        Desired quarter of observation for data products
    month : int or list
        Desired month of observation for data products
    cadence : str
        Desired cadence (`long`, `short`, `any`)
    filetpye : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`)
    """

    def __init__(self, products, filetype):
        self.products = products
        self.filetype = filetype

    def __repr__(self):
        try:
            columns = ['obsID', 'target_name', 'productFilename', 'description', 'distance']
        # some MAST queries do not include a distance column
        except KeyError:
            columns = ['obsID', 'target_name', 'productFilename', 'description']
        return '\n'.join(self.products[columns].pformat(max_width=300))

    @property
    def targets(self):
        """Returns a table of targets and their RA & dec values produced by search"""
        mask = ['target_name', 's_ra', 's_dec']
        return unique(self.products[mask], keys='target_name')

    @property
    def mastID(self):
        """Returns an array of MAST observation IDs"""
        return np.asarray(np.unique(self.products['obsid']), dtype='int')

    @property
    def target_name(self):
        """Returns an array of target names"""
        return self.products['target_name'].data.data

    @property
    def ra(self):
        """Returns an array of RA values for targets in search"""
        return self.products['s_ra'].data.data

    @property
    def dec(self):
        """Returns an array of dec values for targets in search"""
        return self.products['s_dec'].data.data

    def download(self, quality_bitmask='default'):
        """
        Downloads a single KeplerTargetPixelFile or KeplerLightCurveFile object from search result.
        If multiple files are present in `products`, only the first will be downloaded.

        Returns
        -------
        KeplerTargetPixelFile : `KeplerTargetPixelFile` object
            Returns a single `KeplerTargetPixelFile` for first entry in products table
        KeplerLightCurveFile : `KeplerLightCurveFile` object
            Returns a single `KeplerLightCurveFile` for first entry in products table
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
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # check if download directory exists
        if not os.path.isdir(os.path.expanduser('~')+'/.astropy'):
            os.mkdir(os.path.expanduser('~')+'/.astropy')
        # download first product in table
        download_dir = os.path.expanduser('~')+'/.astropy'
        path = Observations.download_products(self.products[:1], mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        if len(self.products) != 1:
            log.warning('Warning: {} files available to download. Only the first file has been '
                        'downloaded. Please use `download_all()` or specify a campaign, quarter, or '
                        'cadence to limit your search.'.format(len(self.products)))

        # return single tpf or lcf
        if self.filetype == "Target Pixel":
            return KeplerTargetPixelFile(path[0],
                                         quality_bitmask=quality_bitmask)
        elif self.filetype == "Lightcurve":
            return KeplerLightCurveFile(path[0],
                                        quality_bitmask=quality_bitmask)

    def download_all(self, quality_bitmask='default'):
        """
        Downloads a KeplerTargetPixelFileCollection or KeplerLightCurveFileCollection from search results.

        Returns
        -------
        KeplerTargetPixelFileCollection : `KeplerTargetPixelFileCollection` object
            Returns a single `KeplerTargetPixelFileCollection` containing all entries in products table
        KeplerLightCurveFileCollection : `KeplerLightCurveFileCollection` object
            Returns a single `KeplerLightCurveFileCollection` containing all entries in products table
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
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # should download to `~/.lightkurve-cache`, make sure dir exists and is accessible
        download_dir = self._fetch_dir()

        # download all products in table
        path = Observations.download_products(self.products, mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        # return collection of tpf or lcf
        if self.filetype == "Target Pixel":
            tpfs = [KeplerTargetPixelFile(p,
                                         quality_bitmask=quality_bitmask) for p in path]
            return TargetPixelFileCollection(tpfs)
        elif self.filetype == "Lightcurve":
            lcs = [KeplerLightCurveFile(p,
                                   quality_bitmask=quality_bitmask) for p in path]
            return LightCurveFileCollection(lcs)

    def _mask_products(self, products, campaign=None, quarter=None, month=None, cadence='long',
                       filetype='Target Pixel', targetlimit=1):
        """
        Masks contents of products table based on given `cadence`, `quarter`, `month`, `campaign`
        constraints.

        Parameters
        ----------
        products : astropy table
            Full astropy table containing data products returned by MAST
        campaign : int or list
            Desired campaign of observation for data products
        quarter : int or list
            Desired quarter of observation for data products
        month : int or list
            Desired month of observation for data products
        cadence : str
            Desired cadence (`long`, `short`, `any`)
        filetpye : str
            Type of files queried at MAST (`Target Pixel` or `Lightcurve`)
        targetlimit : int
            Maximum number of targets in astropy table

        Returns
        -------
        products : astropy table
            Masked astropy table containing desired data products
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

        import pdb; pdb.set_trace()
        if len(np.unique(ids)) < targetlimit:
            log.warning('Target return limit set to {} '
                        'but only {} unique targets found. '
                        'Try increasing the search radius. '
                        ''.format(targetlimit, len(np.unique(ids))))
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

        # If there is nothing in the table, quit now.
        if len(products) == 0:
            return products
        products.sort(['order', 'dates', 'qoc'])

        # For Kepler short cadence data there are additional rules, so find anywhere
        # where there is short cadence data...
        scmask = np.asarray(['Short' in d for d in products['description']]) &\
                 np.asarray(['kplr' in d for d in products['dataURI']])

        if np.any(scmask):
            # If no month is specified, return all
            if self.month is None:
                return products

            self.month = np.atleast_1d(self.month)
            # Get the short cadence date lookup table.
            table = ascii.read(os.path.join(PACKAGEDIR, 'data', 'short_cadence_month_lookup.csv'))
            # Grab the dates of each of the short cadence files. Make sure every entry
            # has the correct month
            finalmask = np.ones(len(products), dtype=bool)
            for c in np.unique(products[scmask]['qoc']):
                ok = (products['qoc'] == c) & (scmask)
                mask = np.zeros(np.shape(products[ok])[0], dtype=bool)
                for m in self.month:
                    udate = (table['StartTime'][np.where(
                        (table['Month'] == m) & (table['Quarter'] == c))[0][0]])
                    mask |= np.asarray(products['dates'][ok]) == udate
                finalmask[ok] = mask
            products = products[finalmask]
            # Sort by id, then date and quarter
            products.sort(['order', 'dates', 'qoc'])
            if len(products) < 1:
                return products

        return products

    def _fetch_dir(self):
        '''
        Checks existance of `~/.lightkurve-cache` directory and creates one if
        none is found.

        Returns
        -------
        download_dir : str
            Path to location of `mastDownload` folder where data downloaded from MAST are stored
        '''
        # check if download directory exists (~/.lightkurve-cache)
        cache_dir = os.path.join(os.path.expanduser('~'), '.lightkurve-cache')
        if os.path.isdir(cache_dir):
            download_dir = cache_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(cache_dir)
                download_dir = cache_dir
            # downloads locally if OS error occurs
            except OSError:
                log.warning('Warning: unable to create .lightkurve-cache directory. '
                            'Downloading MAST files to local directory.')
                download_dir = '.'

        return download_dir


class ArchiveError(Exception):
    """Raised if there is a problem accessing data."""
    pass


def _query_mast(target, cadence='long', radius=.0001, targetlimit=None):
    """
    Returns a table of Kepler or K2 Target Pixel Files or Lightcurve Files
     for a given target.

    Parameters
    ----------
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.
    radius : float
        Search radius in arcseconds
    targetlimit : None or int
        If multiple targets are present within `radius`, limit the number
        of returned TargetPixelFile objects to `targetlimit`.
        If `None`, no limit is applied.

    Returns
    -------
    path : astropy.Table
        Table detailing the available observations on MAST.
    """

    # If passed a SkyCoord, convert it to an RA and Dec
    if isinstance(target, SkyCoord):
        target = '{}, {}'.format(target.ra.deg, target.dec.deg)

    if os.path.exists(str(target)) or str(target).startswith('http'):
        path = [target]

    else:
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

            # query_criteria does not allow a cone search when target_name is passed in
            # so first grab desired target with ~0 arcsecond radius
            target_obs = Observations.query_criteria(target_name=target_name,
                                                     radius='{} deg'.format(.0001),
                                                     project=["Kepler", "K2"],
                                                     obs_collection=["Kepler", "K2"])
            # check if a cone search is being performed
            # if yes, perform a cone search around coordinates of desired target
            if radius < 1:
                path = target_obs
            else:
                ra = target_obs['s_ra'][0]
                dec = target_obs['s_dec'][0]
                path = Observations.query_criteria(coordinates='{} {}'.format(ra, dec),
                                                   radius='{} deg'.format(radius/3600),
                                                   project=["Kepler", "K2"],
                                                   obs_collection=["Kepler", "K2"])
                # if a cone search has been performed, targets will be sorted by `distance`
                path.sort('distance')
        except ValueError:
            # If `target` did not look like a KIC or EPIC ID, then we let MAST
            # resolve the target name to a sky position. Convert radius from arcsec
            # to degrees for query_criteria().
            try:
                path = Observations.query_criteria(objectname=target,
                                                   radius='{} deg'.format(radius/3600),
                                                   project=["Kepler", "K2"],
                                                   obs_collection=["Kepler", "K2"])
            except ResolverError as exc:
                raise ArchiveError(exc)

    # only take the nearest targets up to `targetlimit`
    if targetlimit is not None:
        path = path[:targetlimit]
    return path


def search_targetpixelfile(target, cadence='long', quarter=None, month=None,
                           campaign=None, radius=.0001, targetlimit=1000):

    """
    Fetch a data table for Target Pixel Files within a region of sky. Cone search is
    centered around the position of `target` and extends to a given `radius`.
    If no value is provided for `radius`, only a single target will be returned.

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

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    return search_products(target, filetype="Target Pixel", cadence=cadence,
                           quarter=quarter, month=month, campaign=campaign,
                           radius=radius, targetlimit=targetlimit)


def search_lightcurvefile(target, cadence='long', quarter=None, month=None,
                          campaign=None, radius=.0001, targetlimit=None):

    """
    Fetch a data table for Lightcurve Files within a region of sky. Cone search is
    centered around the position of `target` and extends to a given `radius`.
    If no value is provided for `radius`, only a single target will be returned.

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

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    return search_products(target, filetype="Lightcurve", cadence=cadence,
                           quarter=quarter, month=month, campaign=campaign,
                           radius=radius, targetlimit=targetlimit)


def search_products(target, filetype="Lightcurve", cadence='long', quarter=None, month=None,
                    campaign=None, radius=.0001, targetlimit=1000):
    """Returns a SearchResult object."""
    observations = _query_mast(target, cadence='long', radius=radius, targetlimit=targetlimit)
    products = Observations.get_product_list(observations)
    result = join(products, observations, join_type='left')  # will join on obs_id
    try:
        result.sort(['distance', 'obs_id'])
    except KeyError:
        result.sort('obs_id')
    masked_result = _mask_products(result, filetype=filetype,
                                   campaign=campaign, quarter=quarter,
                                   cadence=cadence, targetlimit=targetlimit)
    return SearchResult(masked_result, filetype)


def _mask_products(products, campaign=None, quarter=None, month=None, cadence='long',
                   filetype='Target Pixel', targetlimit=1):
    """
    Masks contents of products table based on given `cadence`, `quarter`, `month`, `campaign`
    constraints.

    Parameters
    ----------
    products : astropy table
        Full astropy table containing data products returned by MAST
    campaign : int or list
        Desired campaign of observation for data products
    quarter : int or list
        Desired quarter of observation for data products
    month : int or list
        Desired month of observation for data products
    cadence : str
        Desired cadence (`long`, `short`, `any`)
    filetpye : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`)
    targetlimit : int
        Maximum number of targets in astropy table

    Returns
    -------
    products : astropy table
        Masked astropy table containing desired data products
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

    if targetlimit is not None and len(np.unique(ids)) < targetlimit:
        log.warning('Target return limit set to {} '
                    'but only {} unique targets found. '
                    'Try increasing the search radius. '
                    ''.format(targetlimit, len(np.unique(ids))))
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

    # If there is nothing in the table, quit now.
    if len(products) == 0:
        return products
    products.sort(['order', 'dates', 'qoc'])

    # For Kepler short cadence data there are additional rules, so find anywhere
    # where there is short cadence data...
    scmask = np.asarray(['Short' in d for d in products['description']]) &\
             np.asarray(['kplr' in d for d in products['dataURI']])

    if np.any(scmask):
        # If no month is specified, return all
        if month is None:
            return products

        month = np.atleast_1d(month)
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
            return products

    return products
