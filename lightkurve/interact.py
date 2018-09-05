"""Provides tools for interactive visualizations

Example use
-----------

# Must be run from a Jupyter notebook

from lightkurve import KeplerTargetPixelFile
tpf = KeplerTargetPixelFile.from_archive(228682548) # SN 2018 oh for example
tpf.interact()

# An interactive visualization will pop up

"""
from __future__ import division, print_function
import logging
#import numpy as np

__all__ = []


log = logging.getLogger(__name__)


def pixel_selector_standalone(tpf):
    """The standalone version of pixel selector

    Accepts a TargetPixelFile object as an input
    """
    return tpf
