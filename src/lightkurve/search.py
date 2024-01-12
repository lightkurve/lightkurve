"""Defines tools to retrieve Kepler data from the archive at MAST."""
from __future__ import division

import glob
import logging
import os
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Row, Table, join, Column, vstack
from astropy.time import Time
from memoization import cached
from requests import HTTPError

from . import PACKAGEDIR, conf, config
from .collections import LightCurveCollection, TargetPixelFileCollection
from .io import read, AUTHOR_LINKS
from .targetpixelfile import TargetPixelFile
from .utils import (
    LightkurveError,
    LightkurveWarning,
    suppress_stdout,
)

log = logging.getLogger(__name__)

__all__ = [
    "search_targetpixelfile",
    "search_lightcurve",
    "search_tesscut",
    "SearchResult",
]

REPR_COLUMNS_BASE = [
    "#",
    "mission",
    "sequence",
    "author",
    "product_type",
    "exptime",
    "target_name",
    "distance",
    "start_time",
    "end_time",
]


class SearchError(Exception):
    pass


class SearchResult(object):
    """Container for the results returned by the search functions.

    The purpose of this class is to provide a convenient way to inspect and
    download products that have been identified using one of the data search
    functions.

    Parameters
    ----------
    table : `~astropy.table.Table` object
        Astropy table returned by a join of the astroquery `Observations.query_criteria()`
        and `Observations.get_product_list()` methods.
    """

    table = None
    """`~astropy.table.Table` containing the full search results returned by the MAST API."""

    display_extra_columns = []
    """A list of extra columns to be included in the default display of the search result.
    It can be configured in a few different ways.

    For example, to include ``proposal_id`` in the default display, users can set it:

    1. in the user's ``lightkurve.cfg`` file::

        [search]
        # The extra comma at the end is needed for a single extra column
        search_result_display_extra_columns = proposal_id,

    2. at run time::

        import lightkurve as lk
        lk.conf.search_result_display_extra_columns = ['proposal_id']

    3. for a specific `SearchResult` object instance::

        result.display_extra_columns = ['proposal_id']

    See :ref:`configuration <api.config>` for more information.
    """

    def __init__(self, table=None):
        if table is None:
            self.table = Table()
        else:
            self.table = table
            if len(table) > 0:
                self._add_columns()
                self._sort_table()
        self.display_extra_columns = conf.search_result_display_extra_columns

    def _sort_table(self):
        """Sort the table of search results by distance, author, and filename.

        The reason we include "author" in the sort criteria is that Lightkurve v1 only
        showed data products created by the official pipelines (i.e. author equal to
        "Kepler", "K2", or "SPOC"). To maintain backwards compatibility, we want to
        show products from these authors at the top, so that `search.download()`
        operations tend to download the same product in Lightkurve v1 vs v2.
        This ordering is not a judgement on the quality of one product vs another,
        because we love all pipelines!
        """
        sort_priority = {
            "Kepler": 1,
            "K2": 1,
            "SPOC": 1,
            "TESS-SPOC": 2,
            "KBONUS-BKG": 3,
            "QLP": 3,
        }
        self.table["sort_order"] = [
            sort_priority.get(author, 9) for author in self.table["author"]
        ]
        self.table.sort(
            [
                column
                for column in [
                    "distance",
                    "project",
                    "sort_order",
                    "author",
                    "sequence",
                    "start_time",
                    "exptime",
                ]
                if column in self.table.columns
            ]
        )

    def _fix_start_and_end_times(self):
        """The start and stop times for some products are not correct, this function fixes them."""

        # Kepler files have the wrong start and stop times
        kepler_mask = self.table["provenance_name"] == "Kepler"
        filenames = filenames = np.asarray(
            self.table["productFilename"][kepler_mask].data
        )
        start_time = [
            Time(datetime.strptime(filename.split("-")[1].split("_")[0], "%Y%j%H%M%S"))
            for filename in filenames
        ]
        end_time = [
            start_time[idx] + timedelta(days=30)
            if filename.endswith("slc.fits")
            else start_time[idx] + timedelta(days=90)
            for idx, filename in enumerate(filenames)
        ]
        self.table["start_time"][kepler_mask] = start_time
        self.table["end_time"][kepler_mask] = end_time

        # We mask KBONUS times because they are invalid for the quarter data
        if "sequence" in self.table.columns:
            kbonus_mask = self.table["author"] == "KBONUS-BKG"
            kbonus_mask[kbonus_mask] = np.asarray(
                [len(seq) > 0 for seq in self.table["sequence"][kbonus_mask]]
            )
            self.table["start_time"].mask = kbonus_mask
            self.table["end_time"].mask = kbonus_mask

    def _add_columns(self):
        """Adds a user-friendly index (``#``) column and adds column unit
        and display format information.
        """
        self.table = Table(self.table, masked=True, copy=False)
        if "#" not in self.table.columns:
            self.table["#"] = None
        self.table["exptime"].unit = "s"
        self.table["exptime"].format = ".0f"
        self.table["distance"].unit = "arcsec"
        # Some products are HLSPs and some are mission products,
        # this column helps people distinguish them
        self.table["product_type"] = "Mission Product"
        self.table["product_type"][
            ~(self.table["author"] == "SPOC")
            & ~(self.table["author"] == "TESS")
            & ~(self.table["author"] == "K2")
            & ~(self.table["author"] == "Kepler")
        ] = "HLSP"

    def __repr__(self, html=False):
        def to_tess_gi_url(proposal_id):
            if re.match("^G0[12].+", proposal_id) is not None:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs-primary.html#:~:text={proposal_id}"
            elif re.match("^G0[34].+", proposal_id) is not None:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs-em1.html#:~:text={proposal_id}"
            else:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs.html#:~:text={proposal_id}"

        out = "SearchResult containing {} data products.".format(len(self.table))
        if len(self.table) == 0:
            return out
        columns = REPR_COLUMNS_BASE
        if self.display_extra_columns is not None:
            columns = REPR_COLUMNS_BASE + self.display_extra_columns
        # search_tesscut() has fewer columns, ensure we don't try to display columns that do not exist
        columns = [c for c in columns if c in self.table.colnames]

        self.table["#"] = [idx for idx in range(len(self.table))]
        out += "\n\n" + "\n".join(self.table[columns].pformat(max_width=300, html=html))
        # Make sure author names show up as clickable links
        if html:
            for author, url in AUTHOR_LINKS.items():
                out = out.replace(f">{author}<", f"><a href='{url}'>{author}</a><")
            # special HTML formating for TESS proposal_id
            tess_table = self.table[self.table["project"] == "TESS"]
            if "proposal_id" in tess_table.colnames:
                proposal_id_col = np.unique(tess_table["proposal_id"])
            else:
                proposal_id_col = []
            for p_ids in proposal_id_col:
                # for CDIPS products, proposal_id is a np MaskedConstant, not a string
                if p_ids == "N/A" or (not isinstance(p_ids, str)):
                    continue
                # e.g., handle cases with multiple proposals, e.g.,  G12345_G67890
                p_id_links = [
                    f"""\
<a href='{to_tess_gi_url(p_id)}'>{p_id}</a>\
"""
                    for p_id in p_ids.split("_")
                ]
                out = out.replace(f">{p_ids}<", f">{' , '.join(p_id_links)}<")
        return out

    def _repr_html_(self):
        return self.__repr__(html=True)

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
        mask = ["target_name", "s_ra", "s_dec"]
        return Table.from_pandas(
            self.table[mask]
            .to_pandas()
            .drop_duplicates("target_name")
            .reset_index(drop=True)
        )

    @property
    def obsid(self):
        """MAST observation ID for each data product found."""
        return np.asarray(np.unique(self.table["obsid"]), dtype="int64")

    @property
    def ra(self):
        """Right Ascension coordinate for each data product found."""
        return self.table["s_ra"].data.data

    @property
    def dec(self):
        """Declination coordinate for each data product found."""
        return self.table["s_dec"].data.data

    @property
    def mission(self):
        """Kepler quarter or TESS sector names for each data product found."""
        return self.table["mission"].data.data

    @property
    def author(self):
        """Pipeline name for each data product found."""
        return self.table["author"].data.data

    @property
    def target_name(self):
        """Target name for each data product found."""
        return self.table["target_name"].data.data

    @property
    def exptime(self):
        """Exposure time for each data product found."""
        return self.table["exptime"].quantity

    @property
    def distance(self):
        """Distance from the search position for each data product found."""
        return self.table["distance"].quantity

    @property
    def product_type(self):
        """Whether the data product is a mission product or High Level Science Product."""
        return self.table["product_type"].data.data

    @property
    def sequence(self):
        """What quarter, campaign, or sector the data is from."""
        return self.table["sequence"].data.data

    @property
    def start_time(self):
        """Start time of the observation."""
        return Time(self.table["start_time"].data.data)

    @property
    def end_time(self):
        """End time of the observation."""
        return Time(self.table["end_time"].data.data)

    @property
    def year(self):
        """Year of the observation"""
        return list(Time(self.start_time).strftime("%Y"))

    def _download_one(
        self, table, quality_bitmask, download_dir, cutout_size, **kwargs
    ):
        """Private method used by `download()` and `download_all()` to download
        exactly one file from the MAST archive.

        Always returns a `TargetPixelFile` or `LightCurve` object.
        """
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger("astropy").setLevel(log.getEffectiveLevel())

        if download_dir is None:
            download_dir = self._default_download_dir()

        # if the SearchResult row is a TESScut entry, then download cutout
        if "FFI Cutout" in table[0]["description"]:
            try:
                log.debug(
                    "Started downloading TESSCut for '{}' sector {}."
                    "".format(table[0]["target_name"], table[0]["sequence_number"])
                )
                path = self._fetch_tesscut_path(
                    table[0]["target_name"],
                    table[0]["sequence_number"],
                    download_dir,
                    cutout_size,
                )
            except Exception as exc:
                msg = str(exc)
                if "504" in msg:
                    # TESSCut will occasionally return a "504 Gateway Timeout
                    # error" when it is overloaded.
                    raise HTTPError(
                        "The TESS FFI cutout service at MAST appears "
                        "to be temporarily unavailable. It returned "
                        "the following error: {}".format(exc)
                    )
                else:
                    raise SearchError(
                        "Unable to download FFI cutout. Desired target "
                        "coordinates may be too near the edge of the FFI."
                        "Error: {}".format(exc)
                    )

            return read(
                path, quality_bitmask=quality_bitmask, targetid=table[0]["targetid"]
            )

        else:
            if cutout_size is not None:
                warnings.warn(
                    "`cutout_size` can only be specified for TESS "
                    "Full Frame Image cutouts.",
                    LightkurveWarning,
                )
            # Whenever `astroquery.mast.Observations.download_products` is called,
            # a HTTP request will be sent to determine the length of the file
            # prior to checking if the file already exists in the local cache.
            # For performance, we skip this HTTP request and immediately try to
            # find the file in the cache.  The path we check here is consistent
            # with the one hard-coded inside `astroquery.mast.Observations._download_files()`
            # in Astroquery v0.4.1.  It would be good to submit a PR to astroquery
            # so we can avoid having to use this hard-coded hack.
            path = os.path.join(
                download_dir.rstrip("/"),
                "mastDownload",
                table["obs_collection"][0],
                table["obs_id"][0],
                table["productFilename"][0],
            )
            if os.path.exists(path):
                log.debug("File found in local cache.")
            else:
                from astroquery.mast import Observations

                download_url = table[:1]["dataURI"][0]
                log.debug("Started downloading {}.".format(download_url))
                download_response = Observations.download_products(
                    table[:1], mrp_only=False, download_dir=download_dir
                )[0]
                if download_response["Status"] != "COMPLETE":
                    raise LightkurveError(
                        f"Download of {download_url} failed. "
                        f"MAST returns {download_response['Status']}: {download_response['Message']}"
                    )
                path = download_response["Local Path"]
                log.debug("Finished downloading.")
            if table["author"][0] == "KBONUS-BKG":
                quarter = (
                    int(table["sequence"][0].split(" ")[-1])
                    if (len(table["sequence"][0]) != 0)
                    else None
                )
                kwargs["quarter"] = quarter
            return read(path, quality_bitmask=quality_bitmask, **kwargs)

    @suppress_stdout
    def download(
        self, quality_bitmask="default", download_dir=None, cutout_size=None, **kwargs
    ):
        """Download and open the first data product in the search result.

        If multiple files are present in `SearchResult.table`, only the first
        will be downloaded.

        Parameters
        ----------
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` or :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
        download_dir : str, optional
            Location where the data files will be stored.
            If `None` is passed, the value from `cache_dir` configuration parameter is used,
            with "~/.lightkurve/cache" as the default.

            See `~lightkurve.config.get_cache_dir()` for details.
        cutout_size : int, float or tuple, optional
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        kwargs : dict, optional
            Extra keyword arguments passed on to the file format reader function.

        Returns
        -------
        data : `TargetPixelFile` or `LightCurve` object
            The first entry in the products table.

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.

        """
        if len(self.table) == 0:
            warnings.warn(
                "Cannot download from an empty search result.", LightkurveWarning
            )
            return None
        if len(self.table) != 1:
            warnings.warn(
                "Warning: {} files available to download. "
                "Only the first file has been downloaded. "
                "Please use `download_all()` or specify additional "
                "criteria (e.g. quarter, campaign, or sector) "
                "to limit your search.".format(len(self.table)),
                LightkurveWarning,
            )

        return self._download_one(
            table=self.table[:1],
            quality_bitmask=quality_bitmask,
            download_dir=download_dir,
            cutout_size=cutout_size,
            **kwargs,
        )

    @suppress_stdout
    def download_all(
        self, quality_bitmask="default", download_dir=None, cutout_size=None, **kwargs
    ):
        """Download and open all data products in the search result.

        This method will return a `~lightkurve.TargetPixelFileCollection` or
        `~lightkurve.LightCurveCollection`.

        Parameters
        ----------
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` or :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
        download_dir : str, optional
            Location where the data files will be stored.
            If `None` is passed, the value from `cache_dir` configuration parameter is used,
            with "~/.lightkurve/cache" as the default.

            See `~lightkurve.config.get_cache_dir()` for details.
        cutout_size : int, float or tuple, optional
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        kwargs : dict, optional
            Extra keyword arguments passed on to the file format reader function.

        Returns
        -------
        collection : `~lightkurve.collections.Collection` object
            Returns a `~lightkurve.LightCurveCollection` or
            `~lightkurve.TargetPixelFileCollection`,
            containing all entries in the products table

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.
        """
        if len(self.table) == 0:
            warnings.warn(
                "Cannot download from an empty search result.", LightkurveWarning
            )
            return None
        log.debug("{} files will be downloaded.".format(len(self.table)))

        products = []
        for idx in range(len(self.table)):
            products.append(
                self._download_one(
                    table=self.table[idx : idx + 1],
                    quality_bitmask=quality_bitmask,
                    download_dir=download_dir,
                    cutout_size=cutout_size,
                    **kwargs,
                )
            )
        if isinstance(products[0], TargetPixelFile):
            return TargetPixelFileCollection(products)
        else:
            return LightCurveCollection(products)

    def _default_download_dir(self):
        return config.get_cache_dir()

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
        tesscut_dir = os.path.join(download_dir, "tesscut")
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
        sec = TesscutClass().get_sectors(coordinates=coords)
        sector_name = sec[sec["sector"] == sector]["sectorName"][0]
        if isinstance(cutout_size, int):
            size_str = str(int(cutout_size)) + "x" + str(int(cutout_size))
        elif isinstance(cutout_size, tuple) or isinstance(cutout_size, list):
            size_str = str(int(cutout_size[1])) + "x" + str(int(cutout_size[0]))

        # search cache for file with matching ra, dec, and cutout size
        # ra and dec are searched within 0.001 degrees of input target
        ra_string = str(coords.ra.value)
        dec_string = str(coords.dec.value)
        matchstring = r"{}_{}*_{}*_{}_astrocut.fits".format(
            sector_name,
            ra_string[: ra_string.find(".") + 4],
            dec_string[: dec_string.find(".") + 4],
            size_str,
        )
        cached_files = glob.glob(os.path.join(tesscut_dir, matchstring))

        # if any files exist, return the path to them instead of downloading
        if len(cached_files) > 0:
            path = cached_files[0]
            log.debug("Cached file found.")
        # otherwise the file will be downloaded
        else:
            cutout_path = TesscutClass().download_cutouts(
                coordinates=coords, size=cutout_size, sector=sector, path=tesscut_dir
            )
            path = cutout_path[0][0]  # the cutoutpath already contains testcut_dir
            log.debug("Finished downloading.")
        return path


@cached
def search_targetpixelfile(
    target,
    radius=None,
    exptime=None,
    cadence=None,
    mission=("Kepler", "K2", "TESS"),
    author=None,
    quarter=None,
    month=None,
    campaign=None,
    sector=None,
    limit=None,
):
    """Search the `MAST data archive <https://archive.stsci.edu>`_ for target pixel files.

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
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    cadence : 'long', 'short', 'fast', or float
        Synonym for `exptime`. Will likely be deprecated in the future.
    mission : str, tuple of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    author : str, tuple of str, or "any"
        Author of the data product (`provenance_name` in the MAST API).
        Official Kepler, K2, and TESS pipeline products have author names
        'Kepler', 'K2', and 'SPOC'.
        By default, all light curves are returned regardless of the author.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if ``exptime='short'`` you can specify month=1, 2, 3, or 4.
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
        return _search_products(
            target,
            radius=radius,
            filetype="Target Pixel",
            exptime=exptime or cadence,
            mission=mission,
            provenance_name=author,
            quarter=quarter,
            month=month,
            campaign=campaign,
            sector=sector,
            limit=limit,
        )
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


@cached
def search_lightcurve(
    target,
    radius=None,
    exptime=None,
    cadence=None,
    mission=("Kepler", "K2", "TESS"),
    author=None,
    quarter=None,
    month=None,
    campaign=None,
    sector=None,
    limit=None,
):
    """Search the `MAST data archive <https://archive.stsci.edu>`_ for light curves.

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
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    cadence : 'long', 'short', 'fast', or float
        Synonym for `exptime`. This keyword will likely be deprecated in the future.
    mission : str, tuple of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    author : str, tuple of str, or "any"
        Author of the data product (`provenance_name` in the MAST API).
        Official Kepler, K2, and TESS pipeline products have author names
        'Kepler', 'K2', and 'SPOC'.
        Community-provided products that are supported include 'K2SFF', 'EVEREST'.
        By default, all light curves are returned regardless of the author.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if ``exptime='short'`` you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.

    Examples
    --------
    This example demonstrates how to use the `search_lightcurve()` function to
    query and download data. Before instantiating a `LightCurve` object or
    downloading any science products, we can identify potential desired targets with
    `search_lightcurve`::

        >>> from lightkurve import search_lightcurve  # doctest: +SKIP
        >>> search_result = search_lightcurve("Kepler-10")  # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    The above code will query mast for lightcurve files available for the known
    planet system Kepler-10, and display a table containing the available
    data products. Because Kepler-10 was observed in multiple quarters and sectors
    by both Kepler and TESS, the search will return many dozen results.
    If we want to narrow down the search to only return Kepler light curves
    in long cadence, we can use::

        >>> search_result = search_lightcurve("Kepler-10", author="Kepler", exptime=1800)   # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    That is better, we now see 15 light curves corresponding to 15 Kepler quarters.
    If we want to download a `~lightkurve.collections.LightCurveCollection` object containing all
    15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can specify the downloaded products by selecting a specific row using
    rectangular brackets, for example::

        >>> lc = search_result[2].download()  # doctest: +SKIP

    The above line of code will only search and download Quarter 2 data and
    create a `LightCurve` object called lc.

    We can also pass a radius into `search_lightcurve` to perform a cone search::

        >>> search_lightcurve('Kepler-10', radius=100, quarter=4, exptime=1800)  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of
    Kepler-10 and in Quarter 4.  We can then download a
    `~lightkurve.collections.LightCurveFile` containing all these
    light curves using::

        >>> search_lightcurve('Kepler-10', radius=100, quarter=4, exptime=1800).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(
            target,
            radius=radius,
            filetype="Lightcurve",
            exptime=exptime or cadence,
            mission=mission,
            provenance_name=author,
            quarter=quarter,
            month=month,
            campaign=campaign,
            sector=sector,
            limit=limit,
        )
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


@cached
def search_tesscut(target, sector=None):
    """Search the `MAST TESSCut service <https://mast.stsci.edu/tesscut/>`_ for a region
    of sky that is available as a TESS Full Frame Image cutout.

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
        return _search_products(target, filetype="ffi", mission="TESS", sector=sector)
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def _search_products(
    target,
    radius=None,
    filetype="Lightcurve",
    mission=("Kepler", "K2", "TESS"),
    provenance_name=None,
    exptime=(0, 9999),
    quarter=None,
    month=None,
    campaign=None,
    sector=None,
    limit=None,
    **extra_query_criteria,
):
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
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    mission : str, list of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    provenance_name : str, list of str
        Provenance of the data product. Defaults to official products, i.e.
        ('Kepler', 'K2', 'SPOC').  Community-provided products such as 'K2SFF'
        are supported as well.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if ``exptime='short'`` you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    if isinstance(target, int):
        if (0 < target) and (target < 13161030):
            log.warning(
                "Warning: {} may refer to a different Kepler or TESS target. "
                "Please add the prefix 'KIC' or 'TIC' to disambiguate."
                "".format(target)
            )
        elif (0 < 200000000) and (target < 251813739):
            log.warning(
                "Warning: {} may refer to a different K2 or TESS target. "
                "Please add the prefix 'EPIC' or 'TIC' to disambiguate."
                "".format(target)
            )

    # Specifying quarter, campaign, or quarter should constrain the mission
    if quarter is not None:
        mission = "Kepler"
    if campaign is not None:
        mission = "K2"
    if sector is not None:
        mission = "TESS"
    # Ensure mission is a list
    mission = np.atleast_1d(mission).tolist()

    # Avoid filtering on `provenance_name` if `author` equals "any" or "all"
    if provenance_name in ("any", "all") or provenance_name is None:
        provenance_name = None
    else:
        provenance_name = np.atleast_1d(provenance_name).tolist()
    if provenance_name is not None:
        # If author "TESS" is used, we assume it is SPOC
        provenance_name = np.unique(
            [p if p.lower() != "tess" else "SPOC" for p in provenance_name]
        ).tolist()

    # Speed up by restricting the MAST query if we don't want FFI image data
    extra_query_criteria = {}
    if filetype in ["Lightcurve", "Target Pixel"]:
        # At MAST, non-FFI Kepler pipeline products are known as "cube" products,
        # and non-FFI TESS pipeline products are listed as "timeseries".
        extra_query_criteria["dataproduct_type"] = ["cube", "timeseries"]
    # Make sure `search_tesscut` always performs a cone search (i.e. always
    # passed a radius value), because strict target name search does not apply.
    if filetype.lower() == "ffi" and radius is None:
        radius = 0.0001 * u.arcsec
    observations = _query_mast(
        target,
        radius=radius,
        project=mission,
        provenance_name=provenance_name,
        exptime=exptime,
        sequence_number=campaign or sector,
        **extra_query_criteria,
    )

    log.debug(
        "MAST found {} observations. "
        "Now querying MAST for the corresponding data products."
        "".format(len(observations))
    )
    if len(observations) == 0:
        raise SearchError('No data found for target "{}".'.format(target))

    # Light curves and target pixel files
    if filetype.lower() != "ffi":
        from astroquery.mast import Observations

        products = Observations.get_product_list(observations)
        result = join(
            observations,
            products,
            keys="obs_id",
            join_type="right",
            uniq_col_name="{col_name}{table_name}",
            table_names=["", "_products"],
        )
        result.sort(["distance", "obs_id"])
        # Add the user-friendly 'author' column (synonym for 'provenance_name')
        result["author"] = result["provenance_name"]

        # Add the user-friendly 'mission' column
        result["mission"] = result["project"]

        # We need to duplicate any kbonus products because the quarters are in extensions,
        # not separate files.
        kbonus_mask = result["provenance_name"] == "KBONUS-BKG"
        kbonus_tabs = []
        if kbonus_mask.any():
            result["exptime"][kbonus_mask] = 1800
            for kbonus_target in np.unique(result["target_name"][kbonus_mask].data):
                kbonus_tab = vstack(
                    [result[kbonus_mask & (result["target_name"] == kbonus_target)]]
                    * 18
                )
                kbonus_tab["description"] = [
                    f"{desc} - Q{idx}"
                    for idx, desc in enumerate(kbonus_tab["description"])
                ]
                kbonus_tabs.append(kbonus_tab)
            result = vstack([result, vstack(kbonus_tabs)])

        sequence = []
        obs_prefix = {"Kepler": "Quarter", "K2": "Campaign", "TESS": "Sector"}
        for idx in range(len(result)):
            obs_project = result["project"][idx]
            tmp_seqno = result["sequence_number"][idx]
            obs_seqno = f"{tmp_seqno:02d}" if tmp_seqno else ""
            # Kepler sequence_number values were not populated at the time of
            # writing this code, so we parse them from the description field.
            seq = Table.MaskedColumn(result["sequence_number"])
            if obs_project == "Kepler" and seq.mask[idx]:
                try:
                    tmp_seqno = re.findall(r".*Q(\d+)", result["description"][idx])[0]
                    obs_seqno = f"{int(tmp_seqno):02d}"
                except IndexError:
                    obs_seqno = ""
            # K2 campaigns 9, 10, and 11 were split into two sections, which are
            # listed separately in the table with suffixes "a" and "b"
            if obs_project == "K2" and result["sequence_number"][idx] in [9, 10, 11]:
                for half, letter in zip([1, 2], ["a", "b"]):
                    if f"c{tmp_seqno}{half}" in result["productFilename"][idx]:
                        obs_seqno = f"{int(tmp_seqno):02d}{letter}"
            if len(obs_seqno) != 0:
                sequence.append(
                    "{} {}".format(obs_prefix.get(obs_project, ""), obs_seqno)
                )
            else:
                sequence.append("")
        result["sequence"] = np.asarray(sequence)

        masked_result = _filter_products(
            result,
            quarter=quarter,
            exptime=exptime,
            month=month,
            limit=limit,
            filetype=filetype,
        )
        log.debug("MAST found {} matching data products.".format(len(masked_result)))
        masked_result["distance"].info.format = ".1f"  # display <0.1 arcsec

        return SearchResult(masked_result)

    # Full Frame Images
    else:
        cutouts = []
        for idx in np.where(["TESS FFI" in t for t in observations["target_name"]])[0]:
            # if target passed in is a SkyCoord object, convert to RA, dec pair
            if isinstance(target, SkyCoord):
                target = "{}, {}".format(target.ra.deg, target.dec.deg)
            # pull sector numbers
            s = observations["sequence_number"][idx]
            # if the desired sector is available, add a row
            if s in np.atleast_1d(sector) or sector is None:
                cutouts.append(
                    {
                        "description": f"TESS FFI Cutout (sector {s})",
                        "mission": f"TESS Sector {s:02d}",
                        "target_name": str(target),
                        "targetid": str(target),
                        "t_min": observations["t_min"][idx],
                        "t_max": observations["t_max"][idx],
                        "exptime": observations["exptime"][idx],
                        "productFilename": "TESScut",
                        "provenance_name": "TESScut",
                        "author": "TESScut",
                        "distance": 0.0,
                        "sequence_number": s,
                        "project": "TESS",
                        "obs_collection": "TESS",
                    }
                )

        if len(cutouts) > 0:
            log.debug("Found {} matching cutouts.".format(len(cutouts)))
            masked_result = Table(cutouts)
            masked_result.sort(["distance", "sequence_number"])
        else:
            masked_result = None

    if masked_result is not None:
        masked_result["start_time"] = Column(
            Time(masked_result["t_min"] + 2400000.5, format="jd"),
            format=lambda x: f"{x.isot.split('T')[0]}",
        )
        masked_result["end_time"] = Column(
            Time(masked_result["t_max"] + 2400000.5, format="jd"),
            format=lambda x: f"{x.isot.split('T')[0]}",
        )
    return SearchResult(masked_result)


def _query_mast(
    target,
    radius=None,
    project=("Kepler", "K2", "TESS"),
    provenance_name=None,
    exptime=(0, 9999),
    sequence_number=None,
    **extra_query_criteria,
):
    """Helper function which wraps `astroquery.mast.Observations.query_criteria()`
    to return a table of all Kepler/K2/TESS observations of a given target.

    By default only the official data products are returned, but this can be
    adjusted by adding alternative data product names into `provenance_name`.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    project : str, list of str
        Mission name.  Typically 'Kepler', 'K2', or 'TESS'.
        This parameter is case-insensitive.
    provenance_name : str, list of str
        Provenance of the observation.  Common options include 'Kepler', 'K2',
        'SPOC', 'K2SFF', 'EVEREST', 'KEPSEISMIC'.
        This parameter is case-insensitive.
    exptime : (float, float) tuple
        Exposure time range in seconds. Common values include `(59, 61)`
        for Kepler short cadence and `(1799, 1801)` for Kepler long cadence.
    sequence_number : int, list of int
        Quarter, Campaign, or Sector number.
    **extra_query_criteria : kwargs
        Extra criteria to be passed to `astroquery.mast.Observations.query_criteria`.

    Returns
    -------
    obs : astropy.Table
        Table detailing the available observations on MAST.
    """
    # Local astroquery import because the package is not used elsewhere
    from astroquery.exceptions import NoResultsWarning, ResolverError
    from astroquery.mast import Observations

    # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
    if isinstance(target, SkyCoord):
        target = "{}, {}".format(target.ra.deg, target.dec.deg)

    # We pass the following `query_criteria` to MAST regardless of whether
    # we search by position or target name:
    query_criteria = {"project": project, **extra_query_criteria}
    if provenance_name is not None:
        query_criteria["provenance_name"] = provenance_name
    if sequence_number is not None:
        query_criteria["sequence_number"] = sequence_number
    if exptime is not None:
        query_criteria["t_exptime"] = exptime

    # If an exact KIC ID is passed and the author is specified as the mission,
    # we will search by the exact `target_name` under which MAST will know the
    # object to prevent source confusion.
    # For discussion, see e.g. GitHub issues #148, #718.
    exact_target_name = None
    target_lower = str(target).lower()
    # Was a Kepler target ID passed?
    kplr_match = re.match(r"^(kplr|kic) ?(\d+)$", target_lower)
    if kplr_match:
        exact_target_name = f"kplr{kplr_match.group(2).zfill(9)}"
    # Was a K2 target ID passed?
    ktwo_match = re.match(r"^(ktwo|epic) ?(\d+)$", target_lower)
    if ktwo_match:
        exact_target_name = f"ktwo{ktwo_match.group(2).zfill(9)}"
    # Was a TESS target ID passed?
    tess_match = re.match(r"^(tess|tic) ?(\d+)$", target_lower)
    if tess_match:
        exact_target_name = f"{tess_match.group(2).zfill(9)}"

    if provenance_name is not None:
        if len(np.atleast_1d(provenance_name)) == 1:
            mission_match = (
                (
                    bool(kplr_match)
                    & (np.atleast_1d(provenance_name)[0].lower() == "kepler")
                )
                | (
                    bool(ktwo_match)
                    & (np.atleast_1d(provenance_name)[0].lower() == "k2")
                )
                | (
                    bool(tess_match)
                    & (np.atleast_1d(provenance_name)[0].lower() == "spoc")
                )
            )
    else:
        mission_match = False
    # Passed an ID number, no radius, and an official mission author
    # We will do a target name query for the exact target name
    # This is faster than a cone search.
    if exact_target_name and (radius is None) and mission_match:
        log.debug(
            "Started querying MAST for observations with the exact "
            f"target_name='{exact_target_name}'."
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NoResultsWarning)
            warnings.filterwarnings("ignore", message="t_exptime is continuous")
            obs = Observations.query_criteria(
                target_name=exact_target_name, **query_criteria
            )
        if len(obs) > 0:
            # We use `exptime` as an alias for `t_exptime`
            obs["exptime"] = obs["t_exptime"]
            # astroquery does not report distance when querying by `target_name`;
            # we add it here so that the table returned always has this column.
            obs["distance"] = 0.0
            return obs
        else:
            log.debug(f"No observations found. Now performing a cone search instead.")

    # If the above did not return a result, then do a cone search using the MAST name resolver
    # `radius` defaults to 0.0001 and unit arcsecond
    # If radius was originally set to None, we will still need to remove duplicate KIC/EPIC/TIC IDs
    remove_dupes = False
    if radius is None:
        remove_dupes = True
        radius = 0.0001 * u.arcsec
    elif not isinstance(radius, u.quantity.Quantity):
        radius = radius * u.arcsec
    query_criteria["radius"] = str(radius.to(u.deg))

    try:
        log.debug(
            "Started querying MAST for observations within "
            f"{radius.to(u.arcsec)} arcsec of objectname='{target}'."
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NoResultsWarning)
            warnings.filterwarnings("ignore", message="t_exptime is continuous")
            obs = Observations.query_criteria(objectname=target, **query_criteria)
        obs.sort("distance")
        # We use `exptime` as an alias for `t_exptime`
        obs["exptime"] = obs["t_exptime"]
        if remove_dupes & (exact_target_name is not None):
            dupe_mask = ~np.asarray(
                [
                    (target_name[:4] == exact_target_name[:4])
                    & (target_name != exact_target_name)
                    for target_name in obs["target_name"]
                ]
            )
            obs = obs[dupe_mask]
        return obs
    except ResolverError as exc:
        # MAST failed to resolve the object name to sky coordinates
        raise SearchError(exc) from exc


def _filter_products(
    products,
    quarter=None,
    month=None,
    exptime=None,
    limit=None,
    filetype="Target Pixel",
):
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
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    filetype : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`).

    Returns
    -------
    products : `astropy.table.Table` object
        Masked astropy table containing desired data products
    """

    mask = np.ones(len(products), dtype=bool)
    log.debug(f"{mask.sum()} total products found.")
    mask &= _mask_bad_authors(authors=np.asarray(products["author"].data))
    log.debug(f"{mask.sum()} products with valid authors.")
    mask &= _mask_bad_names(filenames=np.asarray(products["productFilename"].data))
    log.debug(f"{mask.sum()} products with valid names.")
    mask &= _mask_by_exptime(products=products, exptime=exptime)
    log.debug(f"{mask.sum()} products with valid exptimes.")
    mask &= _mask_by_filetype(products=products, filetype=filetype)
    log.debug(f"{mask.sum()} products with valid filetypes.")
    mask &= _mask_kepler_products(products=products, quarter=quarter, month=month)
    log.debug(f"{mask.sum()} products with valid Kepler products where applicable.")

    products = products[mask]
    products.sort(["distance", "productFilename"])
    if limit is not None:
        return products[0:limit]
    return products


def _mask_bad_authors(authors):
    """Returns a mask to remove authors we don't have readers for."""
    bad_authors = np.asarray([author not in AUTHOR_LINKS.keys() for author in authors])
    if bad_authors.any():
        log.warn(
            f"Authors {np.unique(authors[bad_authors])} have been removed as `lightkurve` does not have a specific reader for these HLSPs.",
        )
    return ~bad_authors


def _mask_bad_names(filenames):
    """Returns a mask that removes specific files from HLSPs that do not work
    well with the readers, or are redundant."""
    # Allow only fits files
    mask = np.array(
        [
            filename.lower().endswith("fits") or filename.lower().endswith("fits.gz")
            for filename in filenames
        ]
    )
    # We remove one of the TASOC light curve flavors because it is not always present
    # We add a special case for KEPSEISMIC which doesn't obey naming convention
    bad_names = [
        "55d_kepler_v1_cor-filt-inp.fits",
        "80d_kepler_v1_cor-filt-inp.fits",
        "1800_tess_v05_ens-lc.fits",
    ]
    mask &= np.asarray(
        [
            np.all([bad_name not in filename for bad_name in bad_names])
            for filename in filenames
        ]
    )
    return mask


def _mask_by_exptime(products, exptime):
    """Helper function to filter by exposure time."""
    mask = np.ones(len(products), dtype=bool)
    if isinstance(exptime, (int, float)):
        mask &= products["exptime"] == exptime
    elif isinstance(exptime, str):
        exptime = exptime.lower()
        if exptime in ["fast"]:
            mask &= products["exptime"] < 60
        elif exptime in ["short"]:
            mask &= (products["exptime"] >= 60) & (products["exptime"] < 158)
        elif exptime in ["ffi"]:
            mask &= products["exptime"] >= 158
        elif exptime in ["long"]:
            mask &= products["exptime"] >= 300
        elif exptime in ["all", "any"]:
            return mask
        else:
            raise ValueError(f"Can not parse `exptime` `{exptime}`")
    return mask


def _mask_kepler_products(products, quarter=None, month=None):
    """Returns a mask flagging the Kepler products that match the criteria."""
    mask = np.asarray(products["project"].data) != "Kepler"
    if mask.all():
        # No Kepler matches
        return mask

    # Identify quarter by the description.
    # This is necessary because the `sequence_number` field was not populated
    # for Kepler prime data at the time of writing this function.
    if quarter is None:
        quarter = np.arange(18)
    quarter_mask = np.zeros(len(mask), bool)
    for q in np.atleast_1d(quarter):
        quarter_mask |= np.array(
            [
                ((int(seq.split(" ")[-1]) == q) & (seq.lower().startswith("quarter")))
                if len(seq) > 0
                else True
                for seq in products["sequence"].data
            ]
        )
        # If there is no quarter in the sequence, we assume that it is HLSP that covers multiple quarters
    mask |= quarter_mask

    # For Kepler short cadence data the month can be specified
    if month is not None:
        month = np.atleast_1d(month)
        if quarter is None:
            quarter = np.arange(18)
        # Get the short cadence date lookup table.
        table = ascii.read(
            os.path.join(PACKAGEDIR, "data", "short_cadence_month_lookup.csv")
        )
        # The following line is needed for systems where the default integer type
        # is int32 (e.g. Windows/Appveyor), the column will then be interpreted
        # as string which makes the test fail.
        table["StartTime"] = table["StartTime"].astype(str)
        # Grab the dates of each of the short cadence files.
        # Make sure every entry has the correct month
        is_shortcadence = mask & np.asarray(
            ["Short" in desc for desc in products["description"]]
        )
        for idx in np.where(is_shortcadence)[0]:
            try:
                date = (
                    products["dataURI"][idx].split("/")[-1].split("-")[1].split("_")[0]
                )
            except:
                continue

            permitted_dates = []
            for q in np.atleast_1d(quarter):
                for m in np.atleast_1d(month):
                    try:
                        permitted_dates.append(
                            table["StartTime"][
                                np.where(
                                    (table["Month"] == m) & (table["Quarter"] == q)
                                )[0][0]
                            ]
                        )
                    except IndexError:
                        pass
            if not (date in permitted_dates):
                mask[idx] = False
    return mask


def _mask_by_filetype(products, filetype):
    """Helper funtion to mask files that are the wrong filetype"""
    # HLSP products need to be filtered by extension
    if filetype.lower() == "lightcurve":
        mask = np.array(
            [
                filename.lower().endswith("lc.fits")
                for filename in products["productFilename"]
            ]
        )
    elif filetype.lower() == "target pixel":
        mask = np.array(
            [
                filename.lower().endswith(("tp.fits", "targ.fits.gz"))
                for filename in products["productFilename"]
            ]
        )
    elif filetype.lower() == "ffi":
        mask = np.array(["TESScut" in desc for desc in products["description"]])
    return mask


def _resolve_object(target):
    """Ask MAST to resolve an object string to a set of coordinates."""
    from astroquery.mast import MastClass

    # Note: `_resolve_object` was renamed `resolve_object` in astroquery 0.3.10 (2019)
    return MastClass().resolve_object(target)
