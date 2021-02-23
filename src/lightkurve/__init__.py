#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/data/lightkurve.mplstyle".format(PACKAGEDIR)


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
