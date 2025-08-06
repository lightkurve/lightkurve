import warnings
from .prfmodel import *

try:
    from .tpfmodel import *
except ModuleNotFoundError:
    warnings.warn(
        "Warning: the tpfmodel submodule is not available without oktopus installed, "
        "which requires a current version of autograd. See #1452 for details."
    )
