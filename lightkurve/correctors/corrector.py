"""Implements the abstract `Corrector` base class.
"""
from abc import ABC, abstractmethod

import matplotlib
from .. import LightCurve


class Corrector(ABC):
    """Abstract base class documenting the structure of classes intended
    to remove systematic noise from light curves.

    Attributes
    ----------
    lc : LightCurve
        Original, uncorrected light curve, usually assigned by the constructor.
    corrected_lc : LightCurve
        Corrected light curve, assigned by each call to the `correct()` method.

    Methods
    -------
    __init__()
        Accepts all the data required to execute the correction.
    correct() -> LightCurve
        Executes the correction, accepting any optional parameters that
        can be used to modify the way the correction is applied.
    diagnose() -> matplotlib.axes.Axes
        Creates plots to elucidate the user's most recent call to `correct()`.
    """

    lc: LightCurve = None
    corrected_lc: LightCurve = None

    @abstractmethod
    def correct(self) -> LightCurve:
        """Returns a corrected LightCurve, and also stores it as the
        `Corrector.corrected_lc` attribute."""
        pass

    @abstractmethod
    def diagnose(self) -> matplotlib.axes.Axes:
        """Returns plots which elucidate the most recent call to `correct()`."""
        pass
