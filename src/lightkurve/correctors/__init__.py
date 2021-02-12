"""This sub-package defines classes which help remove instrument systematics
or variability from time series photometry data.

Classes provided by this package should inherit from an abstract `Corrector`
class, which provides three key methods::

    Corrector(**data_required):
        .correct(**options) -> Returns a systematics-corrected LightCurve.
        .diagnose(**options) -> Returns figures which elucidate the correction.
        .interact() -> Returns a widget to tune the options interactively (optional).

Classes currently provided are `KeplerCBVCorrector`, `SFFCorrector`, and
`PLDCorrector`.
"""
from .designmatrix import *

from .pldcorrector import *
from .sffcorrector import *
from .cbvcorrector import *
from .regressioncorrector import *
