"""Generic classes and functions which aid the asteroseismology features."""
import numpy as np
import copy
from astropy import units as u
from astropy.units import Quantity

__all__ = ["SeismologyQuantity"]


class SeismologyQuantity(Quantity):
    """Holds an asteroseismic value including its unit, error, and estimation method.

    Compared to a traditional AstroPy `~astropy.units.Quantity` object, this
    class has the following extra attributes:
        * name (e.g. 'deltanu' or 'radius');
        * error (i.e. the uncertainty);
        * method (e.g. specifying the asteroseismic scaling relation);
        * diagnostics;
        * diagnostics_plot_method.
    """

    def __new__(
        cls,
        quantity,
        name=None,
        error=None,
        method=None,
        diagnostics=None,
        diagnostics_plot_method=None,
    ):
        # Note: Quantity is peculiar to sub-class because it inherits from numpy ndarray;
        # see https://docs.astropy.org/en/stable/units/quantity.html#subclassing-quantity.
        self = Quantity.__new__(cls, quantity.value)
        self.__dict__ = quantity.__dict__
        self.name = name
        self.error = error
        self.method = method
        self.diagnostics = diagnostics
        self.diagnostics_plot_method = diagnostics_plot_method
        return self

    def __repr__(self):
        try:
            return "{}: {} {} (method: {})".format(
                self.name, "{:.2f}".format(self.value), self.unit.__str__(), self.method
            )
        except AttributeError:  # Math operations appear to remove Seismic attributes for now
            return super().__repr__()

    def _repr_latex_(self):
        try:
            return "{}: {} {} (method: {})".format(
                self.name,
                "${:.2f}$".format(self.value),
                self.unit._repr_latex_(),
                self.method,
            )
        except AttributeError:  # Math operations appear to remove Seismic attributes for now
            return super()._repr_latex_()


def get_fwhm(periodogram, numax):
    """In a power spectrum of a solar-like oscillator, the power of the
    modes of oscillation will appear in the shape of that looks
    approximately Gaussian, for all basic purposes, also referred to as the
    'mode envelope'. For a given numax (the central frequency of the mode
    envelope), the expected Full Width Half Maximum of the envelope is known
    as a function of numax for evolved Red Giant Branch stars as follows
    (see Mosser et al 2010):

    fwhm = 0.66 * numax^0.88 .

    If the maximum frequency in the periodogram is less than 500 microhertz,
    this function will default to the above equation under the assumption it
    is dealing with an RGB star, which oscillate at lower frequencies.

    If the maximum frequency is above 500 microhertz, the envelope is given
    as a different function of numax (see Lund et al. 2017), as

    fwhm = 0.25 * numax,

    in which case the function assumes it is dealing with a main sequence
    star, which oscillate at higher frequencies.

    Parameters
    ----------
    numax : float
        The estimated position of the numax of the power spectrum. This
        is used to calculated the region autocorrelated with itself.

    Returns
    -------
    fwhm: float
        The estimate full-width-half-maximum of the seismic mode envelope
    """
    # Calculate the index FWHM for a given numax
    if u.Quantity(periodogram.frequency[-1], u.microhertz) > u.Quantity(
        500.0, u.microhertz
    ):
        fwhm = 0.25 * numax
    else:
        fwhm = 0.66 * numax ** 0.88
    return fwhm


def autocorrelate(periodogram, numax, window_width=25.0, frequency_spacing=None):
    """An autocorrelation function (ACF) for seismic mode envelopes.

    We autocorrelate a region with a width of `window_width` (in microhertz)
    around a central frequency `numax` (in microhertz). The window size is
    determined based on the location of the nyquist frequency when
    estimating numax, and based on the expected width of the mode envelope
    of the asteroseismic oscillations when calculating deltanu. The section of
    power being autocorrelated is first resclaed by subtracting its mean, so
    that its noise is centered around zero. If this is not done, noise will
    appear in the ACF as a function of 1/lag.

    Parameters:
    ----------
        numax : float
            The estimated position of the numax of the power spectrum. This
            is used to calculated the region autocorrelated with itself.

        window_width : int or float
            The width of the autocorrelation window around the central
            frequency numax.

        frequency_spacing : float
            The frequency spacing of the periodogram. If none is passed, it
            is calculated internally. This should never be set by the user.

    Returns:
    --------
        acf : array-like
            The autocorrelation power calculated for the given numax
    """
    if frequency_spacing is None:
        frequency_spacing = np.median(np.diff(periodogram.frequency.value))

    spread = int(window_width / 2 / frequency_spacing)  # Find the spread in indices
    x = int(numax / frequency_spacing)  # Find the index value of numax
    x0 = int(
        (periodogram.frequency[0].value / frequency_spacing)
    )  # Transform in case the index isn't from 0
    xt = x - x0
    p_sel = copy.deepcopy(
        periodogram.power[xt - spread : xt + spread].value
    )  # Make the window selection
    p_sel -= np.nanmean(p_sel)  # Make it so that the selection has zero mean.

    C = np.correlate(p_sel, p_sel, mode="full")[
        len(p_sel) - 1 :
    ]  # Correlated the resulting SNR space with itself
    return C
