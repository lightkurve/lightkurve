#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime
import os

try:
    import importlib
except ImportError:
    import imp


PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
if (datetime.now().day == 31) & (datetime.now().month == 10):
    MPLSTYLE = '{}/data/lightkurve_halloween.mplstyle'.format(PACKAGEDIR)
else:
    MPLSTYLE = '{}/data/lightkurve.mplstyle'.format(PACKAGEDIR)

# By default Matplotlib is configured to work with a graphical user interface
# which may require an X11 connection (i.e. a display).  When no display is
# available, errors may occur.  In this case, we default to the robust Agg backend.
# Reference: https://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server
import platform
if platform.system() == "Linux" and os.environ.get('DISPLAY','') == '':
    import matplotlib
    matplotlib.use('Agg')

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
from .seismology import *
from .collections import *
from .search import *

def _whimsify(holiday='halloween'):
    """Add whimsical holiday theme to lightkurve."""
    # Have to set the global paramater for MPLSTYLE back to normal
    global MPLSTYLE
    MPLSTYLE = '{}/data/lightkurve_{}.mplstyle'.format(PACKAGEDIR, holiday.lower())
    if not os.path.isfile(MPLSTYLE):
        raise ValueError('No such holiday as {}'.format(holiday))
    # Modules need to be reloaded
    modules = [lightcurve, lightcurvefile, targetpixelfile, correctors,
                utils, convenience, periodogram, seismology, collections, search]
    try:
        # Python 3.4
        [importlib.reload(module) for module in modules]
    except:
        try:
            # Python <3.3
            [imp.reload(module) for module in modules]
        except:
            # Python 2
            [reload(module) for module in modules]


def dewhimsify():
    """Remove whimsical holiday theme from lightkurve."""
    # Have to set the global paramater for MPLSTYLE back to normal
    global MPLSTYLE
    MPLSTYLE = '{}/data/lightkurve.mplstyle'.format(PACKAGEDIR)

    # Modules need to be reloaded
    modules = [lightcurve, lightcurvefile, targetpixelfile, correctors,
                utils, convenience, periodogram, seismology, collections, search]
    try:
        # Python 3.4
        [importlib.reload(module) for module in modules]
    except:
        try:
            # Python <3.3
            [imp.reload(module) for module in modules]
        except:
            # Python 2
            [reload(module) for module in modules]
