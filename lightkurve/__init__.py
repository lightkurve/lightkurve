#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = '{}/data/lightkurve.mplstyle'.format(PACKAGEDIR)

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from .version import __version__
from .prf import *
from .lightcurve import *
from .lightcurvefile import *
from .correctors import *
from .targetpixelfile import *
from .utils import *
from .convenience import *
from .periodogram import *
from .collections import *
