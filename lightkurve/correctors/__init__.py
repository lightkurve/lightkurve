"""This sub-package defines KeplerCBVCorrector, SFFCorrector, and PLDCorrector.

These classes are intended to help remove instrument systematics or variability
from time series photometry data.
"""
from .pldcorrector import *
from .sffcorrector import *
from .cbvcorrector import *
