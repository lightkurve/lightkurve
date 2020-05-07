"""Defines tools to retrieve Kepler data from the archive at MAST."""
from __future__ import division
import os
import glob
import logging
import re
import warnings
from requests import HTTPError

import numpy as np
from astropy.table import join, Table, Row
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from .targetpixelfile import TargetPixelFile
from .collections import TargetPixelFileCollection, LightCurveFileCollection
from .utils import suppress_stdout, LightkurveWarning, detect_filetype
from . import PACKAGEDIR

log = logging.getLogger(__name__)

__all__ = ['search_targetpixelfile', 'search_lightcurvefile', 'search_tesscut',
           'open', 'SearchResult']


class SearchError(Exception):
    pass


class SearchResult(object):
    """Container for the results returned by `search_targetpixelfile`,
    `search_lightcurvefile`, or `search_tesscut`.

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
            if len(table) > 0:
                self._add_columns()

    def _add_columns(self):
        """Adds user-friendly ``idx`` and ``observation`` columns.

        These columns are not part of the MAST Portal API, but they make the
        display of search results much nicer in Lightkurve.
        """
        self.table['observation'] = None
        self.table['#'] = None
        try:
            prefix = {'Kepler': 'Quarter', 'K2': 'Campaign', 'TESS': 'Sector'}
            for idx in range(len(self.table)):
                self.table['#'][idx] = idx
                mission = self.table['obs_collection'][idx]
                seqno = self.table['sequence_number'][idx]
                if mission == 'Kepler' and self.table['sequence_number'].mask[idx]:
                    seqno = re.findall(r".*Q(\d+)", self.table['description'][idx])[0]
                self.table['observation'][idx] = "{} {} {}".format(mission,
                                                                   prefix[mission],
                                                                   seqno)
        except Exception:
            # be tolerant of any MAST API changes
            # which may cause the code above to fail
            log.warning("Unexpected data encountered in the ``SearchResult`` "
                        "constructor; the MAST API may have changed.")

    def __repr__(self, html=False):
        out = 'SearchResult containing {} data products.'.format(len(self.table))
        if len(self.table) == 0:
            return out
        columns = ['#', 'observation', 'target_name', 'productFilename', 'distance']
        return out + '\n\n' + '\n'.join(self.table[columns].pformat(max_width=300, html=html))

    def _repr_html_(self):
        return self.__repr__(html=True)

    def __getitem__(self, key):
        """Implements indexing and slicing, e.g. SearchResult[2:5]."""
        # this check is necessary due to an astropy bug
        # for more information, see issue #445
        if key == -1:
            key = len(self.table) - 1
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
        return np.asarray(np.unique(self.table['obsid']), dtype='int64')

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

    def _download_one(self, table, quality_bitmask, download_dir, cutout_size):
        """Private method used by `download()` and `download_all()` to download
        exactly one file from the MAST archive.

        Always returns a `TargetPixelFile` or `LightCurveFile` object.
        """
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger('astropy').setLevel(log.getEffectiveLevel())

        if download_dir is None:
            download_dir = self._default_download_dir()

        # if the SearchResult row is a TESScut entry, then download cutout
        if 'FFI Cutout' in table[0]['description']:
            try:
                log.debug("Started downloading TESSCut for '{}' sector {}."
                          "".format(table[0]['target_name'], table[0]['sequence_number']))
                path = self._fetch_tesscut_path(table[0]['target_name'],
                                                table[0]['sequence_number'],
                                                download_dir,
                                                cutout_size)
            except Exception as exc:
                msg = str(exc)
                if "504" in msg:
                    # TESSCut will occasionally return a "504 Gateway Timeout
                    # error" when it is overloaded.
                    raise HTTPError('The TESS FFI cutout service at MAST appears '
                                    'to be temporarily unavailable. It returned '
                                    'the following error: {}'.format(exc))
                else:
                    raise SearchError('Unable to download FFI cutout. Desired target '
                                    'coordinates may be too near the edge of the FFI.'
                                    'Error: {}'.format(exc))

            return _open_downloaded_file(path,
                                         quality_bitmask=quality_bitmask,
                                         targetid=table[0]['targetid'])

        else:
            if cutout_size is not None:
                warnings.warn('`cutout_size` can only be specified for TESS '
                              'Full Frame Image cutouts.', LightkurveWarning)
            from astroquery.mast import Observations
            log.debug("Started downloading {}.".format(table[:1]['dataURL'][0]))
            path = Observations.download_products(table[:1], mrp_only=False,
                                                  download_dir=download_dir)['Local Path'][0]
            log.debug("Finished downloading.")
            # open() will determine filetype and return
            return _open_downloaded_file(path, quality_bitmask=quality_bitmask)

    @suppress_stdout
    def download(self, quality_bitmask='default', download_dir=None, cutout_size=None):
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
        cutout_size : int, float or tuple
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)

        Returns
        -------
        data : `TargetPixelFile` or `LightCurveFile` object
            The first entry in the products table.

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.
        """
        if len(self.table) == 0:
            warnings.warn("Cannot download from an empty search result.",
                          LightkurveWarning)
            return None
        if len(self.table) != 1:
            warnings.warn('Warning: {} files available to download. '
                          'Only the first file has been downloaded. '
                          'Please use `download_all()` or specify additional '
                          'criteria (e.g. quarter, campaign, or sector) '
                          'to limit your search.'.format(len(self.table)),
                          LightkurveWarning)

        return self._download_one(table=self.table[:1],
                                  quality_bitmask=quality_bitmask,
                                  download_dir=download_dir,
                                  cutout_size=cutout_size)

    @suppress_stdout
    def download_all(self, quality_bitmask='default', download_dir=None, cutout_size=None):
        """Returns a `~lightkurve.collections.TargetPixelFileCollection` or
        `~lightkurve.collections.LightCurveFileCollection`.

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
        cutout_size : int, float or tuple
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)

        Returns
        -------
        collection : `~lightkurve.collections.Collection` object
            Returns a `~lightkurve.collections.LightCurveFileCollection` or
            `~lightkurve.collections.TargetPixelFileCollection`,
            containing all entries in the products table

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.
        """
        if len(self.table) == 0:
            warnings.warn("Cannot download from an empty search result.",
                          LightkurveWarning)
            return None
        log.debug("{} files will be downloaded.".format(len(self.table)))

        products = []
        for idx in range(len(self.table)):
            products.append(self._download_one(table=self.table[idx:idx+1],
                                               quality_bitmask=quality_bitmask,
                                               download_dir=download_dir,
                                               cutout_size=cutout_size))
        if isinstance(products[0], TargetPixelFile):
            return TargetPixelFileCollection(products)
        else:
            return LightCurveFileCollection(products)

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

    def _fetch_tesscut_path(self, target, sector, download_dir, cutout_size):
        """Downloads TESS FFI cutout and returns path to local file.

        Parameters
        ----------
        download_dir : str
            Path to location of `.lightkurve-cache` directory where downloaded
            cutouts are stored
        cutout_size : int, float or tuple
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)

        Returns
        -------
        path : str
            Path to locally downloaded cutout file
        """
        from astroquery.mast import TesscutClass
        coords = _resolve_object(target)

        # Set cutout_size defaults
        if cutout_size is None:
            cutout_size = 5

        # Check existence of `~/.lightkurve-cache/tesscut`
        tesscut_dir = os.path.join(download_dir, 'tesscut')
        if not os.path.isdir(tesscut_dir):
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(tesscut_dir)
            # downloads into default cache if OSError occurs
            except OSError:
                tesscut_dir = download_dir

        # Resolve SkyCoord of given target
        coords = _resolve_object(target)

        # build path string name and check if it exists
        # this is necessary to ensure cutouts are not downloaded multiple times
        sec = TesscutClass().get_sectors(coords)
        sector_name = sec[sec['sector'] == sector]['sectorName'][0]
        if isinstance(cutout_size, int):
            size_str = str(int(cutout_size)) + 'x' + str(int(cutout_size))
        elif isinstance(cutout_size, tuple) or isinstance(cutout_size, list):
            size_str = str(int(cutout_size[1])) + 'x' + str(int(cutout_size[0]))

        # search cache for file with matching ra, dec, and cutout size
        # ra and dec are searched within 0.001 degrees of input target
        ra_string = str(coords.ra.value)
        dec_string = str(coords.dec.value)
        matchstring = r"{}_{}*_{}*_{}_astrocut.fits".format(sector_name,
                                                            ra_string[:ra_string.find('.')+4],
                                                            dec_string[:dec_string.find('.')+4],
                                                            size_str)
        cached_files = glob.glob(os.path.join(tesscut_dir, matchstring))

        # if any files exist, return the path to them instead of downloading
        if len(cached_files) > 0:
            path = cached_files[0]
            log.debug("Cached file found.")
        # otherwise the file will be downloaded
        else:
            cutout_path = TesscutClass().download_cutouts(coords, size=cutout_size,
                                                          sector=sector, path=tesscut_dir)
            path = os.path.join(download_dir, cutout_path[0][0])
            log.debug("Finished downloading.")
        return path


def search_targetpixelfile(target, radius=None, cadence='long',
                           mission=('Kepler', 'K2', 'TESS'), quarter=None,
                           month=None, campaign=None, sector=None, limit=None):
    """Searches the `public data archive at MAST <https://archive.stsci.edu>`_
    for a Kepler or TESS `~lightkurve.targetpixelfile.TargetPixelFile`.

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
    mission : str, list of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if cadence='short' you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.

    Examples
    --------
    This example demonstrates how to use the `search_targetpixelfile()` function
    to query and download data. Before instantiating a
    `~lightkurve.targetpixelfile.KeplerTargetPixelFile` object or
    downloading any science products, we can identify potential desired targets
    with `search_targetpixelfile()`::

        >>> search_result = search_targetpixelfile('Kepler-10')  # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    The above code will query mast for Target Pixel Files (TPFs) available for
    the known planet system Kepler-10, and display a table containing the
    available science products. Because Kepler-10 was observed during 15 Quarters,
    the table will have 15 entries. To obtain a
    `~lightkurve.collections.TargetPixelFileCollection` object containing all
    15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can download a single product by limiting our search::

        >>> tpf = search_targetpixelfile('Kepler-10', quarter=2).download()  # doctest: +SKIP

    The above line of code will only download Quarter 2 and create a single
    `~lightkurve.targetpixelfile.KeplerTargetPixelFile` object called `tpf`.

    We can also pass a radius into `search_targetpixelfile` to perform a cone search::

        >>> search_targetpixelfile('Kepler-10', radius=100).targets  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of Kepler-10.
    We can download a `~lightkurve.collections.TargetPixelFileCollection` object
    containing all available products for these targets in Quarter 4 with::

        >>> search_targetpixelfile('Kepler-10', radius=100, quarter=4).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(target, radius=radius, filetype="Target Pixel",
                                cadence=cadence, mission=mission, quarter=quarter,
                                month=month, campaign=campaign, sector=sector, limit=limit)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def search_lightcurvefile(target, radius=None, cadence='long',
                          mission=('Kepler', 'K2', 'TESS'), quarter=None,
                          month=None, campaign=None, sector=None, limit=None):
    """Searches the `public data archive at MAST <https://archive.stsci.edu>`_ for a Kepler or TESS
    :class:`LightCurveFile <lightkurve.lightcurvefile.LightCurveFile>`.

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
    mission : str, list of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if cadence='short' you can specify month=1, 2, 3, or 4.
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
    `~lightkurve.collections.LightCurveFileCollection` object containing all
    15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can specify the downloaded products by limiting our search::

        >>> lcf = search_lightcurvefile('Kepler-10', quarter=2).download()  # doctest: +SKIP

    The above line of code will only search and download Quarter 2 data and
    create a `LightCurveFile` object called lcf.

    We can also pass a radius into `search_lightcurvefile` to perform a cone search::

        >>> search_lightcurvefile('Kepler-10', radius=100, quarter=4)  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of
    Kepler-10 and in Quarter 4.  We can then download a
    `~lightkurve.collections.LightCurveFileCollection` containing all these
    products using::

        >>> search_lightcurvefile('kepler-10', radius=100, quarter=4).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(target, radius=radius, filetype="Lightcurve",
                                cadence=cadence, mission=mission, quarter=quarter,
                                month=month, campaign=campaign, sector=sector, limit=limit)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def search_tesscut(target, sector=None):
    """Searches MAST for TESS Full Frame Image cutouts containing a desired target or region.

    This feature uses the `TESScut service <https://mast.stsci.edu/tesscut/>`_
    provided by the TESS data archive at MAST.  If you use this service in
    your work, please `cite TESScut <https://ascl.net/code/v/2239>`_ in your
    publications.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:

            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    sector : int or list
        TESS Sector number. Default (None) will return all available sectors. A
        list of desired sectors can also be provided.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.
    """
    try:
        return _search_products(target, filetype="ffi", mission='TESS', sector=sector)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def _search_products(target, radius=None, filetype="Lightcurve", cadence='long',
                     mission=('Kepler', 'K2', 'TESS'), quarter=None, month=None,
                     campaign=None, sector=None, limit=None, **extra_query_criteria):
    """Helper function which returns a SearchResult object containing MAST
    products that match several criteria.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    filetype : {'Target pixel', 'Lightcurve', 'FFI'}
        Type of files queried at MAST.
    cadence : str
        Desired cadence (`long`, `short`, `any`)
    mission : str, list of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if cadence='short' you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    if isinstance(target, int):
        if (0 < target) and (target < 13161030):
            log.warning("Warning: {} may refer to a different Kepler or TESS target. "
                        "Please add the prefix 'KIC' or 'TIC' to disambiguate."
                        "".format(target))
        elif (0 < 200000000) and (target < 251813739):
            log.warning("Warning: {} may refer to a different K2 or TESS target. "
                        "Please add the prefix 'EPIC' or 'TIC' to disambiguate."
                        "".format(target))

    # Speed up by restricting the MAST query if we don't want FFI image data
    extra_query_criteria = {}
    if filetype in ['Lightcurve', 'Target Pixel']:
        # At MAST, non-FFI Kepler pipeline products are known as "cube" products,
        # and non-FFI TESS pipeline products are listed as "timeseries".
        extra_query_criteria['dataproduct_type'] = ['cube', 'timeseries']
    observations = _query_mast(target, project=mission, radius=radius, **extra_query_criteria)
    log.debug("MAST found {} observations. "
              "Now querying MAST for the corresponding data products."
              "".format(len(observations)))
    if len(observations) == 0:
        raise SearchError('No data found for target "{}".'.format(target))

    # Light curves and target pixel files
    if filetype.lower() != 'ffi':
        from astroquery.mast import Observations
        products = Observations.get_product_list(observations)
        result = join(products, observations, keys="obs_id", join_type='left',
                      uniq_col_name='{col_name}{table_name}', table_names=['', '_2'])
        result.sort(['distance', 'obs_id'])

        masked_result = _filter_products(result, filetype=filetype,
                                         campaign=campaign, quarter=quarter,
                                         cadence=cadence, project=mission,
                                         month=month, sector=sector, limit=limit)
        log.debug("MAST found {} matching data products.".format(len(masked_result)))
        return SearchResult(masked_result)

    # Full Frame Images
    else:
        cutouts = []
        for idx in np.where(['TESS FFI' in t for t in observations['target_name']])[0]:
            # if target passed in is a SkyCoord object, convert to RA, dec pair
            if isinstance(target, SkyCoord):
                target = '{}, {}'.format(target.ra.deg, target.dec.deg)
            # pull sector numbers
            s = observations['sequence_number'][idx]
            # if the desired sector is available, add a row
            if s in np.atleast_1d(sector) or sector is None:
                cutouts.append({'description': 'TESS FFI Cutout (sector {})'.format(s),
                                'target_name': str(target),
                                'targetid': str(target),
                                'productFilename': 'TESSCut',
                                'distance': 0.0,
                                'sequence_number': s,
                                'obs_collection': 'TESS'}
                               )
        if len(cutouts) > 0:
            log.debug("Found {} matching cutouts.".format(len(cutouts)))
            masked_result = Table(cutouts)
            masked_result.sort(['distance', 'sequence_number'])
        else:
            masked_result = None
        return SearchResult(masked_result)


def _query_mast(target, radius=None, project=('Kepler', 'K2', 'TESS'), **extra_query_criteria):
    """Helper function which wraps `astroquery.mast.Observations.query_criteria()`
    to returns a table of all Kepler or K2 observations of a given target.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    project : str, list of str
        'Kepler', 'K2', and/or 'TESS'.

    Returns
    -------
    obs : astropy.Table
        Table detailing the available observations on MAST.
    """
    # If passed a SkyCoord, convert it to an RA and Dec
    if isinstance(target, SkyCoord):
        target = '{}, {}'.format(target.ra.deg, target.dec.deg)

    project = np.atleast_1d(project)

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
        with warnings.catch_warnings():
            # suppress misleading AstropyWarning
            warnings.simplefilter('ignore', AstropyWarning)
            from astroquery.mast import Observations
            log.debug("Started querying MAST for observations within {} of target_name='{}'."
                      "".format(radius.to(u.arcsec), target_name))
            target_obs = Observations.query_criteria(target_name=target_name,
                                                     radius=str(radius.to(u.deg)),
                                                     project=project,
                                                     obs_collection=project,
                                                     **extra_query_criteria)

        if len(target_obs) == 0:
            raise ValueError("No observations found for '{}'.".format(target_name))

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
            with warnings.catch_warnings():
                # suppress misleading AstropyWarning
                warnings.simplefilter('ignore', AstropyWarning)
                from astroquery.mast import Observations
                log.debug("Started querying MAST for observations within {} of coordinates='{} {}'."
                          "".format(radius.to(u.arcsec), ra, dec))
                obs = Observations.query_criteria(coordinates='{} {}'.format(ra, dec),
                                                  radius=str(radius.to(u.deg)),
                                                  project=project,
                                                  obs_collection=project,
                                                  **extra_query_criteria)
            obs.sort('distance')
        return obs
    except ValueError:
        pass

    # If `target` did not look like a KIC or EPIC ID, then we let MAST
    # resolve the target name to a sky position. Convert radius from arcsec
    # to degrees for query_criteria().
    from astroquery.exceptions import ResolverError
    try:
        with warnings.catch_warnings():
            # suppress misleading AstropyWarning
            warnings.simplefilter('ignore', AstropyWarning)
            from astroquery.mast import Observations
            log.debug("Started querying MAST for observations within {} of objectname='{}'."
                      "".format(radius.to(u.arcsec), target))
            obs = Observations.query_criteria(objectname=target,
                                              radius=str(radius.to(u.deg)),
                                              project=project,
                                              obs_collection=project,
                                              **extra_query_criteria)
        obs.sort('distance')
        return obs
    except ResolverError as exc:
        raise SearchError(exc)


def _filter_products(products, campaign=None, quarter=None, month=None,
                     sector=None, cadence='long', limit=None,
                     project=('Kepler', 'K2', 'TESS'), filetype='Target Pixel'):
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
    project = np.atleast_1d(project)
    project_lower = [p.lower() for p in project]

    mask = np.zeros(len(products), dtype=bool)

    if 'kepler' in project_lower and campaign is None and sector is None:
        mask |= _mask_kepler_products(products, quarter=quarter, month=month,
                                      cadence=cadence, filetype=filetype)
    if 'k2' in project_lower and quarter is None and sector is None:
        mask |= _mask_k2_products(products, campaign=campaign,
                                  cadence=cadence, filetype=filetype)
    if 'tess' in project_lower and quarter is None and campaign is None:
        mask |= _mask_tess_products(products, sector=sector, filetype=filetype)

    products = products[mask]

    products.sort(['distance', 'productFilename'])
    if limit is not None:
        return products[0:limit]
    return products


def _mask_kepler_products(products, quarter=None, month=None, cadence='long',
                          filetype='Target Pixel'):
    """Returns a mask flagging the Kepler products that match the criteria."""
    mask = np.array([proj.lower() == 'kepler' for proj in products['project']])
    if mask.sum() == 0:
        return mask

    # Allow only fits files
    mask &= np.array([uri.lower().endswith('fits') or
                      uri.lower().endswith('fits.gz')
                      for uri in products['productFilename']])

    # Filters on cadence and product type
    if cadence in ['short', 'sc']:
        description_string = "{} Short".format(filetype)
    elif cadence in ['any', 'both']:
        description_string = "{}".format(filetype)
    else:
        description_string = "{} Long".format(filetype)
    mask &= np.array([description_string in desc for desc in products['description']])

    # Identify quarter by the description.
    if quarter is not None:
        quarter_mask = np.zeros(len(products), dtype=bool)
        for q in np.atleast_1d(quarter):
            quarter_mask |= np.array([desc.lower().replace('-', '').endswith('q{}'.format(q))
                                      for desc in products['description']])
        mask &= quarter_mask

    # For Kepler short cadence data the month can be specified
    if month is not None:
        month = np.atleast_1d(month)
        # Get the short cadence date lookup table.
        table = ascii.read(os.path.join(PACKAGEDIR, 'data', 'short_cadence_month_lookup.csv'))
        # The following line is needed for systems where the default integer type
        # is int32 (e.g. Windows/Appveyor), the column will then be interpreted
        # as string which makes the test fail.
        table['StartTime'] = table['StartTime'].astype(str)
        # Grab the dates of each of the short cadence files.
        # Make sure every entry has the correct month
        is_shortcadence = mask & np.asarray(['Short' in desc for desc in products['description']])
        for idx in np.where(is_shortcadence)[0]:
            quarter = int(products['description'][idx].split(' - ')[-1][1:].replace('-', ''))
            date = products['dataURI'][idx].split('/')[-1].split('-')[1].split('_')[0]
            permitted_dates = []
            for m in month:
                try:
                    permitted_dates.append(table['StartTime'][
                        np.where((table['Month'] == m) & (table['Quarter'] == quarter))[0][0]
                                    ])
                except IndexError:
                    pass
            if not (date in permitted_dates):
                mask[idx] = False

    return mask


def _mask_k2_products(products, campaign=None, cadence='long', filetype='Target Pixel'):
    """Returns a mask flagging the K2 products that match the criteria."""
    mask = np.array([proj.lower() == 'k2' for proj in products['project']])
    if mask.sum() == 0:
        return mask

    # Allow only fits files
    mask &= np.array([uri.lower().endswith('fits') or
                      uri.lower().endswith('fits.gz')
                      for uri in products['productFilename']])

    # Filters on cadence and product type
    if cadence in ['short', 'sc']:
        description_string = "{} Short".format(filetype)
    elif cadence in ['any', 'both']:
        description_string = "{}".format(filetype)
    else:
        description_string = "{} Long".format(filetype)
    mask &= np.array([description_string in desc for desc in products['description']])

    # Identify campaign by the description.
    if campaign is not None:
        campaign_mask = np.zeros(len(products), dtype=bool)
        for c in np.atleast_1d(campaign):
            campaign_mask |= np.array(['c{:02d}'.format(c) in desc.lower().replace('-', '') or
                                       'c{:03d}'.format(c) in desc.lower().replace('-', '')
                                       for desc in products['description']])
        mask &= campaign_mask

    return mask


def _mask_tess_products(products, sector=None, filetype='Target Pixel'):
    """Returns a mask flagging the TESS products that match the criteria."""
    mask = np.array([p.lower() == 'spoc' for p in products['project']])
    if mask.sum() == 0:
        return mask

    # Allow only fits files
    mask &= np.array([uri.lower().endswith('fits') or
                      uri.lower().endswith('fits.gz')
                      for uri in products['productFilename']])

    # Filter on product type
    if filetype.lower() == 'lightcurve':
        description_string = 'Light curves'
    elif filetype.lower() == 'target pixel':
        description_string = 'Target pixel files'
    elif filetype.lower() == 'ffi':
        description_string = 'TESScut'
    mask &= np.array([description_string in desc for desc in products['description']])

    # Identify sector by the description.
    if sector is not None:
        sector_mask = np.zeros(len(products), dtype=bool)
        for s in np.atleast_1d(sector):
            sector_mask |= np.array([("-" in fn) and (fn.split('-')[1] == 's{:04d}'.format(s))
                                     for fn in products['productFilename']])
        mask &= sector_mask

    return mask


def open(path_or_url, **kwargs):
    """Opens any valid Kepler or TESS data file and returns an instance of
    `~lightkurve.lightcurvefile.LightCurveFile` or
    `~lightkurve.targetpixelfile.TargetPixelFile`.

    This function will use the `detect_filetype()` function to
    automatically detect the type of the data product, and return the
    appropriate object. File types currently supported are::

        * `KeplerTargetPixelFile` (typical suffix "-targ.fits.gz");
        * `KeplerLightCurveFile` (typical suffix "llc.fits");
        * `TessTargetPixelFile` (typical suffix "_tp.fits");
        * `TessLightCurveFile` (typical suffix "_lc.fits").

    Parameters
    ----------
    path_or_url : str
        Path or URL of a FITS file.

    Returns
    -------
    data : a subclass of  `~lightkurve.targetpixelfile.TargetPixelFile` or
        `~lightkurve.lightcurvefile.LightCurveFile`, depending on the detected file type.

    Raises
    ------
    ValueError : raised if the data product is not recognized as a Kepler or
        TESS product.

    Examples
    --------
    To open a target pixel file using its path or URL, simply use:

        >>> tpf = open("mytpf.fits")  # doctest: +SKIP
    """
    log.debug("Opening {}.".format(path_or_url))
    # pass header into `detect_filetype()`
    try:
        with fits.open(path_or_url) as temp:
            filetype = detect_filetype(temp[0].header)
            log.debug("Detected filetype: '{}'.".format(filetype))
    except OSError as e:
        filetype = None
        # Raise an explicit FileNotFoundError if file not found
        if 'No such file' in str(e):
            raise e

    # if the filetype is recognized, instantiate a class of that name
    if filetype is not None:
        return getattr(__import__('lightkurve'), filetype)(path_or_url, **kwargs)
    else:
        # if these keywords don't exist, raise `ValueError`
        raise ValueError("Not recognized as a Kepler or TESS data product: "
                         "{}".format(path_or_url))


def _open_downloaded_file(path, **kwargs):
    """Wrapper around `open()` which yields a more clear error message when
    the file was downloaded from MAST but corrupted, e.g. due to the
    download having been interrupted."""
    try:
        return open(path, **kwargs)
    except ValueError:
        raise SearchError("Failed to open the downloaded file ({}). "
                          "The file was likely only partially downloaded. "
                          "Please remove it from your disk and try again.".format(path))


def _resolve_object(target):
    """Ask MAST to resolve an object string to a set of coordinates."""
    from astroquery.mast import MastClass
    # `_resolve_object` was renamed `resolve_object` in astroquery 0.3.10 (2019)
    try:
        return MastClass().resolve_object(target)
    except AttributeError:
        return MastClass()._resolve_object(target)
