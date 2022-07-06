#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/data/lightkurve.mplstyle".format(PACKAGEDIR)
"""Lightkurve's stylesheet for matplotlib.

It is useful for users who create their own figures and want their figures following
Lightkurve's style.

    Examples
    --------
    Create a scatter plot with a custom size using Lightkurve's style.

        >>> with plt.style.context(MPLSTYLE):  # doctest: +SKIP
        >>>     ax = plt.figure(figsize=(6, 3)).gca()  # doctest: +SKIP
        >>>     lc.scatter(ax=ax)  # doctest: +SKIP

"""

# Bibtex entry detailing how to cite the package
__citation__ = """@MISC{2018ascl.soft12013L,
    author = {{Lightkurve Collaboration} and {Cardoso}, J.~V.~d.~M. and
                {Hedges}, C. and {Gully-Santiago}, M. and {Saunders}, N. and
                {Cody}, A.~M. and {Barclay}, T. and {Hall}, O. and
                {Sagear}, S. and {Turtelboom}, E. and {Zhang}, J. and
                {Tzanidakis}, A. and {Mighell}, K. and {Coughlin}, J. and
                {Bell}, K. and {Berta-Thompson}, Z. and {Williams}, P. and
                {Dotson}, J. and {Barentsen}, G.},
    title = "{Lightkurve: Kepler and TESS time series analysis in Python}",
    keywords = {Software, NASA},
howpublished = {Astrophysics Source Code Library},
        year = 2018,
    month = dec,
archivePrefix = "ascl",
    eprint = {1812.013},
    adsurl = {http://adsabs.harvard.edu/abs/2018ascl.soft12013L},
}"""


import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from . import config as _config


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `lightkurve`.

    Refer to `astropy.config.ConfigNamespace` for API details.

    Refer to `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html#accessing-values>`_
    for usage.

    The attributes listed below are the available configuration parameters.

    Attributes
    ----------
    search_result_display_extra_columns
        List of extra columns to be included when displaying a SearchResult object.

    cache_dir
        Default cache directory for data files downloaded, etc. Defaults to ``~/.lightkurve/cache`` if not specified.

    warn_legacy_cache_dir
        If set to True, issue warning if the legacy default cache directory exists. Default is True.
    """
    # Note: when using list or string_list datatype,
    # the behavior of astropy's parsing of the config file value:
    # - it does not accept python list literal
    # - it accepts a comma-separated list of string
    #   - for a single value, it needs to be ended with a comma
    # see: https://configobj.readthedocs.io/en/latest/configobj.html#the-config-file-format
    search_result_display_extra_columns = _config.ConfigItem(
        [],
        "List of extra columns to be included when displaying a SearchResult object.",
        cfgtype="string_list",
        module="lightkurve.search"
    )

    cache_dir = _config.ConfigItem(
        None,
        "Default cache directory for data files downloaded, etc.",
        cfgtype="string",
        module="lightkurve.config"
    )

    warn_legacy_cache_dir = _config.ConfigItem(
        True,
        "If set to True, issue warning if the legacy default cache directory exists.",
        cfgtype="boolean",
        module="lightkurve.config"
    )

conf = Conf()


from .version import __version__
from . import units  # enable ppt and ppm as units
from .time import *
from .lightcurve import *
from .lightcurvefile import *
from .correctors import *
from .targetpixelfile import *
from .utils import *
from .convenience import *
from .collections import *
from .io import *
from .search import *

from . import config
config.warn_if_default_cache_dir_migration_needed()
