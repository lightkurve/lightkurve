"""Defines tools to retrieve Kepler data from the archive at MAST."""
from __future__ import division
import os
import logging
import numpy as np
import warnings

from astropy.table import join, Table, Row
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy import units as u

from astroquery.mast import Observations
from astroquery.exceptions import ResolverError

from .lightcurvefile import KeplerLightCurveFile, TessLightCurveFile
from .targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from .collections import TargetPixelFileCollection, LightCurveFileCollection
from .utils import suppress_stdout, LightkurveWarning
from . import PACKAGEDIR

log = logging.getLogger(__name__)

__all__ = ['search_targetpixelfile', 'search_lightcurvefile', 'open']


class SearchError(Exception):
    pass


class SearchResult(object):
    """Container for the results returned by `search_targetpixelfile` or
    `search_lightcurvefile`.

    The purpose of this class is to provide a convenient way to inspect and
    download products that have been identified using one of the data search
    functions.

    Parameters
    ----------
    table : `astropy.table.Table` object
        Astropy table returned by a join of the astroquery `Observations.query_criteria()`
        and `Observations.get_product_list()` methods.
    """
    def __init__(self, table=None):
        if table is None:
            self.table = Table()
        else:
            self.table = table

    def __repr__(self):
        out = 'SearchResult containing {} data products.'.format(len(self.table))
        if len(self.table) == 0:
            return out
        columns = ['obsID', 'target_name', 'productFilename', 'description', 'distance']
        return out + '\n\n' + '\n'.join(self.table[columns].pformat(max_width=300))

    def __getitem__(self, key):
        """Implements indexing and slicing, e.g. SearchResult[2:5]."""
        selection = self.table[key]
        # Indexing a Table with an integer will return a Row
        if isinstance(selection, Row):
            selection = Table(selection)
        return SearchResult(table=selection)

    def __len__(self):
        """Returns the number of products in the SearchResult table."""
        return len(self.table)

    @property
    def unique_targets(self):
        """Returns a table of targets and their RA & dec values produced by search"""
        mask = ['target_name', 's_ra', 's_dec']
        return Table.from_pandas(self.table[mask].to_pandas().drop_duplicates('target_name').reset_index(drop=True))

    @property
    def obsid(self):
        """Returns an array of MAST observation IDs"""
        return np.asarray(np.unique(self.table['obsid']), dtype='int')

    @property
    def target_name(self):
        """Returns an array of target names"""
        return self.table['target_name'].data.data

    @property
    def ra(self):
        """Returns an array of RA values for targets in search"""
        return self.table['s_ra'].data.data

    @property
    def dec(self):
        """Returns an array of dec values for targets in search"""
        return self.table['s_dec'].data.data

    @suppress_stdout
    def download(self, quality_bitmask='default', download_dir=None):
        """Returns a single `KeplerTargetPixelFile` or `KeplerLightCurveFile` object.

        If multiple files are present in `SearchResult.table`, only the first
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
        if len(self.table) == 0:
            warnings.warn("Cannot download from an empty search result.",
                          LightkurveWarning)
            return None
        if len(self.table) != 1:
            warnings.warn('Warning: {} files available to download. '
                          'Only the first file has been downloaded. '
                          'Please use `download_all()` or specify a campaign, quarter, or '
                          'cadence to limit your search.'.format(len(self.table)),
                          LightkurveWarning)

        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # download first product in table
        if download_dir is None:
            download_dir = self._default_download_dir()

        path = Observations.download_products(self.table[:1], mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        # return single tpf or lcf
        tpf_files = ['lpd-targ.fits', 'spd-targ.fits']
        lcf_files = ['llc.fits', 'slc.fits']
        if any(file in self.table['productFilename'][0] for file in tpf_files):
            return KeplerTargetPixelFile(path[0], quality_bitmask=quality_bitmask)
        elif any(file in self.table['productFilename'][0] for file in lcf_files):
            return KeplerLightCurveFile(path[0], quality_bitmask=quality_bitmask)

    @suppress_stdout
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
        if len(self.table) == 0:
            warnings.warn("Cannot download from an empty search result.",
                          LightkurveWarning)
            return None

        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        # download all products listed in self.products
        if download_dir is None:
            download_dir = self._default_download_dir()

        path = Observations.download_products(self.table, mrp_only=False,
                                              download_dir=download_dir)['Local Path']

        # return collection of tpf or lcf
        tpf_files = ['lpd-targ.fits', 'spd-targ.fits']
        lcf_files = ['llc.fits', 'slc.fits']
        if any(file in self.table['productFilename'][0] for file in tpf_files):
            tpfs = [KeplerTargetPixelFile(p, quality_bitmask=quality_bitmask)
                    for p in path]
            return TargetPixelFileCollection(tpfs)
        elif any(file in self.table['productFilename'][0] for file in lcf_files):
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


def search_targetpixelfile(target, radius=None, cadence='long', quarter=None,
                           month=None, campaign=None, limit=None):
    """Searches MAST for Target Pixel Files.

    This function fetches a data table that lists the Target Pixel Files (TPFs)
    that fall within a region of sky centered around the position of `target`
    and within a cone of a given `radius`. If no value is provided for `radius`,
    only a single target will be returned.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:

            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    cadence : str
        'long' or 'short'.
    quarter, campaign : int, list of ints
        Kepler Quarter or K2 Campaign number.
        By default all quarters and campaigns will be returned.
    month : 1, 2, 3, or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if cadence='short' you need to specify month=1, 2, or 3.
        By default all months will be returned.
    limit : int
        Maximum number of products to return.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.

    Examples
    --------
    This example demonstrates how to use the `search_targetpixelfile()` function to
    query and download data. Before instantiating a `KeplerTargetPixelFile` object or
    downloading any science products, we can identify potential desired targets with
    `search_targetpixelfile()`::

        >>> search_result = search_targetpixelfile('Kepler-10')  # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    The above code will query mast for Target Pixel Files (TPFs) available for
    the known planet system Kepler-10, and display a table containing the
    available science products. Because Kepler-10 was observed during 15 Quarters,
    the table will have 15 entries. If we want to download a
    `TargetPixelFileCollection` object containing all 15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can download a single product by limiting our search::

        >>> lcf = search_targetpixelfile('Kepler-10', quarter=2).download()  # doctest: +SKIP

    The above line of code will only download Quarter 2 and create a single
    `KeplerTargetPixelFile` object called lcf.

    We can also pass a radius into `search_targetpixelfile` to perform a cone search::

        >>> search_targetpixelfile('Kepler-10', radius=100).targets  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of Kepler-10.
    We can download a `TargetPixelFileCollection` object containing all available products
    for these targets in Quarter 4 with::

        >>> search_targetpixelfile('Kepler-10', radius=100, quarter=4).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(target, radius=radius, filetype="Target Pixel",
                                cadence=cadence, quarter=quarter, month=month,
                                campaign=campaign, limit=limit)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def search_lightcurvefile(target, radius=None, cadence='long', quarter=None,
                          month=None, campaign=None, limit=None):
    """Returns a SearchResult with MAST LightCurveFiles which match the criteria.

    This function fetches a data table that lists the Light Curve Files
    that fall within a region of sky centered around the position of `target`
    and within a cone of a given `radius`. If no value is provided for `radius`,
    only a single target will be returned.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:

            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    cadence : str
        'long' or 'short'.
    quarter, campaign : int, list of ints
        Kepler Quarter or K2 Campaign number.
        By default all quarters and campaigns will be returned.
    month : 1, 2, 3, or list of int
        For Kepler's prime mission, there are three short-cadence
        LightCurveFiles for each quarter, each covering one month.
        Hence, if cadence='short' you need to specify month=1, 2, or 3.
        By default all months will be returned.
    limit : int
        Maximum number of products to return.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.

    Examples
    --------
    This example demonstrates how to use the `search_lightcurvefile()` function to
    query and download data. Before instantiating a `KeplerLightCurveFile` object or
    downloading any science products, we can identify potential desired targets with
    `search_lightcurvefile`::

        >>> from lightkurve import search_lightcurvefile  # doctest: +SKIP
        >>> search_result = search_lightcurvefile("Kepler-10")  # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    The above code will query mast for lightcurve files available for the known
    planet system Kepler-10, and display a table containing the available
    data products. Because Kepler-10 was observed in 15 quarters, the search
    result will list 15 different files. If we want to download a
    `LightCurveFileCollection` object containing all 15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can specify the downloaded products by limiting our search::

        >>> lcf = search_lightcurvefile('Kepler-10', quarter=2).download()  # doctest: +SKIP

    The above line of code will only search and download Quarter 2 data and
    create a `LightCurveFile` object called lcf.

    We can also pass a radius into `search_lightcurvefile` to perform a cone search::

        >>> search_lightcurvefile('Kepler-10', radius=100, quarter=4)  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of
    Kepler-10 and in Quarter 4.  We can then download a `LightCurveFileCollection`
    containing all these products using::

        >>> search_lightcurvefile('kepler-10', radius=100, quarter=4).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(target, radius=radius, filetype="Lightcurve",
                                cadence=cadence, quarter=quarter, month=month,
                                campaign=campaign, limit=limit)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def _search_products(target, radius=None, filetype="Lightcurve", cadence='long',
                     quarter=None, month=None, campaign=None, limit=None):
    """Helper function which returns a SearchResult object containing MAST
    products that match several criteria.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
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
    limit : int
        Maximum number of products to return

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    observations = _query_mast(target, cadence='long', radius=radius)
    if len(observations) == 0:
        raise SearchError('No data found for target "{}."'.format(target))
    products = Observations.get_product_list(observations)
    result = join(products, observations, join_type='left')  # will join on obs_id
    result.sort(['distance', 'obs_id'])

    masked_result = _filter_products(result, filetype=filetype, campaign=campaign,
                                     quarter=quarter, cadence=cadence, month=month,
                                     limit=limit)
    return SearchResult(masked_result)


def _query_mast(target, radius=None, cadence='long'):
    """Helper function which wraps `astroquery.mast.Observations.query_criteria()`
    to returns a table of all Kepler or K2 observations of a given target.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    cadence: 'short' or 'long'
        Specify short (1-min) or long (30-min) cadence data.

    Returns
    -------
    obs : astropy.Table
        Table detailing the available observations on MAST.
    """
    # If passed a SkyCoord, convert it to an RA and Dec
    if isinstance(target, SkyCoord):
        target = '{}, {}'.format(target.ra.deg, target.dec.deg)

    if radius is None:
        radius = .0001 * u.arcsec
    elif not isinstance(radius, u.quantity.Quantity):
        radius = radius * u.arcsec

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
                                                 radius=str(radius.to(u.deg)),
                                                 project=["Kepler", "K2"],
                                                 obs_collection=["Kepler", "K2"])
        if len(target_obs) == 0:
            raise ValueError("No observations found for {}".format(target_name))

        # check if a cone search is being performed
        # if yes, perform a cone search around coordinates of desired target
        if radius < (0.1 * u.arcsec):
            obs = target_obs
            # astroquery does not return distance if target_name is given;
            # we add it here so that the table returned always has this column.
            obs['distance'] = 0.
        else:
            ra = target_obs['s_ra'][0]
            dec = target_obs['s_dec'][0]
            obs = Observations.query_criteria(coordinates='{} {}'.format(ra, dec),
                                              radius=str(radius.to(u.deg)),
                                              project=["Kepler", "K2"],
                                              obs_collection=["Kepler", "K2"])
    except ValueError:
        # If `target` did not look like a KIC or EPIC ID, then we let MAST
        # resolve the target name to a sky position. Convert radius from arcsec
        # to degrees for query_criteria().
        try:
            obs = Observations.query_criteria(objectname=target,
                                              radius=str(radius.to(u.deg)),
                                              project=["Kepler", "K2"],
                                              obs_collection=["Kepler", "K2"])
        except ResolverError as exc:
            raise SearchError(exc)

    obs.sort('distance')  # ensure table returned is sorted by distance
    return obs


def _filter_products(products, campaign=None, quarter=None, month=None,
                     cadence='long', filetype='Target Pixel', limit=None):
    """Helper function which filters a SearchResult's products table by one or
    more criteria.

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
    filetype : str
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


def open(path_or_url):
    """Opens a Kepler or TESS data product.

    This function will automatically detect the type of the data product,
    and return the appropriate object. File types currently supported are::

        * `KeplerTargetPixelFile` (typical suffix "-targ.fits.gz");
        * `KeplerLightCurveFile` (typical suffix "llc.fits");
        * `TessTargetPixelFile` (typical suffix "_tp.fits");
        * `TessLightCurveFile` (typical suffix "_lc.fits").

    The function will detect the file type by looking at both the TELESCOP and
    CREATOR keywords in the first extension of the FITS file.
    If the file is not recognized as a Kepler or TESS data product,
    a `ValueError` is raised.

    Parameters
    ----------
    path_or_url : str
        Path or URL of a FITS file.

    Returns
    -------
    data : a subclass of :class:`TargetPixelFile` or :class:`LightCurveFile`,
        depending on the detected file type.

    Raises
    ------
    ValueError : raised if the data product is not recognized as a Kepler or
        TESS product.

    Examples
    --------
    To open a target pixel file using its path or URL, simply use:

        >>> tpf = open("mytpf.fits")  # doctest: +SKIP
    """
    hdulist = fits.open(path_or_url)
    try:
        # use `telescop` keyword to determine mission
        # and `creator` to determine tpf or lc
        telescop = hdulist[0].header['telescop']
        creator = hdulist[0].header['creator']
        if telescop == 'Kepler':
            if 'TargetPixel' in creator:
                return KeplerTargetPixelFile(path_or_url)
            elif 'Flux' in creator:
                return KeplerLightCurveFile(path_or_url)
        elif telescop == 'TESS':
            if 'TargetPixel' in creator:
                return TessTargetPixelFile(path_or_url)
            elif 'Flux' in creator:
                return TessLightCurveFile(path_or_url)
    # if these keywords don't exist, raise `ValueError`
    except KeyError:
        pass
    raise ValueError('Given fits file not recognized as Kepler or TESS observation.')
