"""Defines tools to retrieve Kepler data from the archive at MAST."""
from __future__ import division
import os
import logging
import numpy as np

from astropy.table import unique, join, Table, Row
from astropy.coordinates import SkyCoord
from astropy.io import ascii

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError

from .lightcurvefile import KeplerLightCurveFile
from .targetpixelfile import KeplerTargetPixelFile
from .collections import TargetPixelFileCollection, LightCurveFileCollection
from . import PACKAGEDIR

log = logging.getLogger(__name__)

__all__ = ['search_targetpixelfile', 'search_lightcurvefile']


class ArchiveError(Exception):
    """Raised if there is a problem accessing data."""
    pass


class SearchResult(object):
    """Holds results returned by `search_targetpixelfile` or `search_lightcurvefile`.

    The purpose of this class is to provide a convenient way to inspect and
    download products that have been identified using one of the data search
    functions.

    Parameters
    ----------
    products : `astropy.table.Table` object
        Astropy table returned by a join of the astroquery `Observations.query_criteria()`
        and `Observations.get_product_list()` methods.
    """
    def __init__(self, products):
        self.products = products

    def __repr__(self):
        columns = ['obsID', 'target_name', 'productFilename', 'description', 'distance']
        return '\n'.join(self.products[columns].pformat(max_width=300))

    def __getitem__(self, key):
        products_slice = self.products[key]
        # Indexing a Table with a single integer will return a Row
        if isinstance(products_slice, Row):
            products_slice = Table(products_slice)
        return SearchResult(products=products_slice)

    def __len__(self):
        return len(self.products)

    @property
    def unique_targets(self):
        """Returns a table of targets and their RA & dec values produced by search"""
        mask = ['target_name', 's_ra', 's_dec']
        return unique(self.products[mask], keys='target_name')

    @property
    def obsid(self):
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

    def download(self, quality_bitmask='default', download_dir=None):
        """Returns a single `KeplerTargetPixelFile` or `KeplerLightCurveFile` object.

        If multiple files are present in `SearchResult.products`, only the first
        will be downloaded.

        Parameters
        ----------
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
        download_dir : str
            Location where the data files will be stored.
            Defaults to "~/.lightkurve-cache" if `None` is passed.

        Returns
        -------
        data : `TargetPixelFile` or `LightCurveFile` object
            The first entry in the products table.
        """
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # download first product in table
        if download_dir is None:
            download_dir = self._default_download_dir()
        path = Observations.download_products(self.products[:1], mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        if len(self.products) != 1:
            log.warning('Warning: {} files available to download. '
                        'Only the first file has been downloaded. '
                        'Please use `download_all()` or specify a campaign, quarter, or '
                        'cadence to limit your search.'.format(len(self.products)))

        # return single tpf or lcf
        tpf_files = ['lpd-targ.fits', 'spd-targ.fits']
        lcf_files = ['llc.fits', 'slc.fits']
        if any(file in self.products['productFilename'][0] for file in tpf_files):
            return KeplerTargetPixelFile(path[0], quality_bitmask=quality_bitmask)
        elif any(file in self.products['productFilename'][0] for file in lcf_files):
            return KeplerLightCurveFile(path[0], quality_bitmask=quality_bitmask)

    def download_all(self, quality_bitmask='default', download_dir=None):
        """Returns a `TargetPixelFileCollection or `LightCurveFileCollection`.

         Parameters
         ----------
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
        download_dir : str
            Location where the data files will be stored.
            Defaults to "~/.lightkurve-cache" if `None` is passed.

        Returns
        -------
        collection : `lightkurve.Collection` object
            Returns a `LightCurveFileCollection`  or `TargetPixelFileCollection`,
            containing all entries in the products table
        """
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # download all products listed in self.products
        if download_dir is None:
            download_dir = self._default_download_dir()
        path = Observations.download_products(self.products, mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        # return collection of tpf or lcf
        tpf_files = ['lpd-targ.fits', 'spd-targ.fits']
        lcf_files = ['llc.fits', 'slc.fits']
        if any(file in self.products['productFilename'][0] for file in tpf_files):
            tpfs = [KeplerTargetPixelFile(p, quality_bitmask=quality_bitmask)
                    for p in path]
            return TargetPixelFileCollection(tpfs)
        elif any(file in self.products['productFilename'][0] for file in lcf_files):
            lcs = [KeplerLightCurveFile(p, quality_bitmask=quality_bitmask)
                   for p in path]
            return LightCurveFileCollection(lcs)

    def _default_download_dir(self):
        """Returns the default path to the directory where files will be downloaded.

        By default, this method will return "~/.lightkurve-cache" and create
        this directory if it does not exist.  If the directory cannot be
        access or created, then it returns the local directory (".").

        Returns
        -------
        download_dir : str
            Path to location of `mastDownload` folder where data downloaded from MAST are stored
        """
        download_dir = os.path.join(os.path.expanduser('~'), '.lightkurve-cache')
        if os.path.isdir(download_dir):
            return download_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(download_dir)
            # downloads locally if OS error occurs
            except OSError:
                log.warning('Warning: unable to create {}. '
                            'Downloading MAST files to the current '
                            'working directory instead.'.format(download_dir))
                download_dir = '.'

        return download_dir


def _query_mast(target, cadence='long', radius=.0001):
    """Returns a table of all Kepler or K2 observations of a given target.

    This function wraps the `astroquery.mast.Observations.query_criteria()`
    method.

    Parameters
    ----------
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.
    radius : float
        Search radius in arcseconds

    Returns
    -------
    obs : astropy.Table
        Table detailing the available observations on MAST.
    """

    # If passed a SkyCoord, convert it to an RA and Dec
    if isinstance(target, SkyCoord):
        target = '{}, {}'.format(target.ra.deg, target.dec.deg)

    try:
        # If `target` looks like a KIC or EPIC ID, we will pass the exact
        # `target_name` under which MAST will know the object to prevent
        # source confusion (see GitHub issue #148).
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
        if radius < 0.1:
            obs = target_obs
            # astroquery does not return distance if target_name is given;
            # we add it here so that the table returned always has this column.
            obs['distance'] = 0.
        else:
            ra = target_obs['s_ra'][0]
            dec = target_obs['s_dec'][0]
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
        except ResolverError as exc:
            raise ArchiveError(exc)

    obs.sort('distance')  # ensure table returned is sorted by distance
    return obs


def search_targetpixelfile(target, cadence='long', quarter=None, month=None,
                           campaign=None, radius=.0001, limit=None):

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
                           radius=radius, limit=limit)


def search_lightcurvefile(target, cadence='long', quarter=None, month=None,
                          campaign=None, radius=.0001, limit=None):

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
                           radius=radius, limit=limit)


def search_products(target, filetype="Lightcurve", cadence='long', quarter=None, month=None,
                    campaign=None, radius=.0001, limit=None):
    """Returns a SearchResult object.

    Parameters
    ----------
    target : str or int
        KIC/EPIC ID or object name.
    filetype : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`)
    cadence : str
        Desired cadence (`long`, `short`, `any`)
    quarter : int or list
        Desired quarter of observation for data products
    month : int or list
        Desired month of observation for data products
    campaign : int or list
        Desired campaign of observation for data products
    radius : float
        Search radius in arcseconds
    limit : int
        Maximum number of products to return

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    observations = _query_mast(target, cadence='long', radius=radius)
    products = Observations.get_product_list(observations)
    result = join(products, observations, join_type='left')  # will join on obs_id
    result.sort(['distance', 'obs_id'])

    masked_result = _filter_products(result, filetype=filetype, campaign=campaign,
                                     quarter=quarter, cadence=cadence, limit=limit)
    return SearchResult(masked_result)


def _filter_products(products, campaign=None, quarter=None, month=None,
                     cadence='long', filetype='Target Pixel', limit=None):
    """Returns a products table filtered by one or more criteria.

    This function can filter based on `cadence`, `quarter`, `month`, `campaign`
    constraints.

    Parameters
    ----------
    products : `astropy.table.Table` object
        Astropy table containing data products returned by MAST
    campaign : int or list
        Desired campaign of observation for data products
    quarter : int or list
        Desired quarter of observation for data products
    month : int or list
        Desired month of observation for data products
    cadence : str
        Desired cadence (`long`, `short`, `any`)
    filetpye : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`).

    Returns
    -------
    products : `astropy.table.Table` object
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

    # If there is nothing in the table, quit now.
    if len(products) == 0:
        return products
    products.sort(['distance', 'dates', 'qoc'])

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
        products.sort(['distance', 'dates', 'qoc'])

    if limit is not None:
        return products[0:limit]
    return products
