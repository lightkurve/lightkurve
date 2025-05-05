import warnings
from .prfmodel import *

try:
    from .tpfmodel import *
except ModuleNotFoundError:
    warnings.warn(
        "Warning: the tpfmodel module is not available without oktopus. See lightkurve PR #1452 for details."
    )
