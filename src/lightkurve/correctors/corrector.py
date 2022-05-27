"""Implements the abstract `Corrector` base class.
"""
from abc import ABC, abstractmethod

import matplotlib
import numpy as np

from .. import LightCurve
from .metrics import overfit_metric_lombscargle, underfit_metric_neighbors


class Corrector(ABC):
    """Abstract base class documenting the required structure of classes
    designed to remove systematic noise from light curves.

    Attributes
    ----------
    original_lc : LightCurve
        The uncorrected light curve.  Must be passed into (or computed by) the
        constructor method.
    corrected_lc : LightCurve
        Corrected light curve. Must be updated upon each call to the `correct()` method.
    cadence_mask : np.array of dtype=bool
        Boolean array with the same length as `original_lc`.
        True indicates that a cadence should be used to fit the noise model.
        By setting certain cadences to False, users can exclude those cadences
        from informing the noise model, which will help prevent the overfitting
        of those signals (e.g. exoplanet transits).
        By default, the cadence mask is True across all cadences.

    Methods
    -------
    __init__()
        Accepts all the data required to execute the correction.
        The constructor must set the `original_lc` attribute.
    correct() -> LightCurve
        Executes the correction, optionally accepting meaningful parameters that
        can be used to modify the way the correction is applied.
        This method must set or update the `corrected_lc` attribute on each run.
    diagnose() -> matplotlib.axes.Axes
        Creates plots to elucidate the user's most recent call to `correct()`.
    """

    @property
    def original_lc(self) -> LightCurve:
        if hasattr(self, "_original_lc"):
            return self._original_lc
        else:
            raise AttributeError("`original_lc` has not been instantiated yet.")

    @original_lc.setter
    def original_lc(self, original_lc):
        self._original_lc = original_lc

    @property
    def corrected_lc(self) -> LightCurve:
        if hasattr(self, "_corrected_lc"):
            return self._corrected_lc
        else:
            raise AttributeError(
                "You need to call the `correct()` method "
                "before you can access `corrected_lc`."
            )

    @corrected_lc.setter
    def corrected_lc(self, corrected_lc):
        self._corrected_lc = corrected_lc

    @property
    def cadence_mask(self) -> np.array:
        if not hasattr(self, "_cadence_mask"):
            self._cadence_mask = np.ones(len(self.original_lc), dtype=bool)
        return self._cadence_mask

    @cadence_mask.setter
    def cadence_mask(self, cadence_mask):
        self._cadence_mask = cadence_mask

    def __init__(self, original_lc: LightCurve) -> None:
        """Constructor method.

        The constructor shall:

        * accept all data required to run the correction (e.g. light curves,
          target pixel files, engineering data).
        * instantiate the `original_lc` property.
        """
        self.original_lc = original_lc

    @abstractmethod
    def correct(
        self, cadence_mask: np.array = None, optimize: bool = False
    ) -> LightCurve:
        """Returns a `LightCurve` from which systematic noise has been removed.

        This method shall:

        * accept meaningful parameters that can be used to tune the correction,
          including:

          * `optimize`: should an optimizer be used to tune the parameters?
          * `cadence_mask`: flags cadences to be used to fit the noise model.

        * store all parameters as object attributes (e.g. `self.optimize`, `self.cadence_mask`);
        * store helpful diagnostic information as object attributes;
        * store the result in the `self.corrected_lc` attribute;
        * return `self.corrected_lc`.
        """
        if cadence_mask:
            self.cadence_mask = cadence_mask
        # ... perform correction ...
        # self.corrected_lc = corrected_lc
        # return corrected_lc

    @abstractmethod
    def diagnose(self) -> matplotlib.axes.Axes:
        """Returns plots which elucidate the most recent call to `correct()`.

        This method shall plot useful diagnostic information which have been
        stored as object attributes during the most recent call to `correct()`.
        """
        pass

    def compute_overfit_metric(self, **kwargs) -> float:
        """Measures the degree of over-fitting in the correction.

        See the docstring of `lightkurve.correctors.metrics.overfit_metric_lombscargle`
        for details.

        Returns
        -------
        overfit_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """
        return overfit_metric_lombscargle(
            # Ignore masked cadences in the computation
            self.original_lc[self.cadence_mask],
            self.corrected_lc[self.cadence_mask],
            **kwargs
        )

    def compute_underfit_metric(self, **kwargs) -> float:
        """Measures the degree of under-fitting the correction.

        See the docstring of `lightkurve.correctors.metrics.underfit_metric_neighbors`
        for details.

        Returns
        -------
        underfit_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """
        return underfit_metric_neighbors(self.corrected_lc[self.cadence_mask], **kwargs)
