import numpy as np
from astropy import units as u
from astropy.stats import LombScargle
from matplotlib import pyplot as plt

__all__ = ['Periodogram']


class Periodogram(object):
    """Represents a power spectrum.

    Attributes
    ----------
    frequencies : array-like
        List of frequencies.
    powers : array-like
        Power measurements.
    """
    def __init__(self, frequencies=None, powers=None):
        self.frequencies = frequencies
        self.powers = powers

    @staticmethod
    def from_lightcurve(lc, frequencies=None):
        """Creates a Periodogram object from a LightCurve instance using
        the Lomb-Scargle method.

        By default, the periodogram will be created for a regular grid of
        frequencies from 1 microhertz to the Nyquist frequency.  Alternatively,
        the user can provide a custom regular grid using the `frequencies`
        parameter.

        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.  In the future, we should use the
        lc.time_format attribute to verify this.

        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        frequencies : array-like
            The regular grid of frequencies to use.  The frequencies must be
            in units microhertz.  Alternatively, an AstroPy Quantity object can
            be passed with any unit of type '1/time'.

        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        if frequencies is None:
            nyquist_frequency = 0.5 * (1./((np.median(lc.time[1:] - lc.time[0:-1])*u.day).to(u.second))).to(u.microhertz).value
            frequencies = np.linspace(1, nyquist_frequency, len(lc.time) // 2) * u.microhertz
        if not isinstance(frequencies, u.Quantity):
            frequencies = u.Quantity(frequencies, u.microhertz)
        lombscargle = LombScargle((lc.time * u.day).to(u.second), lc.flux * 1e6)
        uHz_conv = 1./(((1./u.day).to(u.microhertz)))
        powers = lombscargle.power(frequencies, method="fast", normalization="psd")
        powers *= uHz_conv / len(lc.time)  # Convert to ppm^2/uHz
        return Periodogram(frequencies=frequencies, powers=powers)

    def plot(self, frequency=None, scale="linear", ax=None, numax=None, **kwargs):
        """Plots the periodogram.

        Parameters
        ----------
        frequency: array-like
            Over what frequencies (in microhertz) will periodogram plot
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        numax: bool
            Plot the numax value as well?
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.

        """
        if ax is None:
            fig, ax = plt.subplots()
        # Plot frequency and power
        ax.plot(self.frequencies, self.powers, **kwargs)
        ax.set_xlabel("Frequency [$\mu$Hz]")
        ax.set_ylabel("Power [ppm$^2$/$\mu$Hz]")
        if numax:
            ax.fill_between([numax.value*0.8, numax.value*1.2],
                            self.powers.value.min(),
                            self.powers.value.max(),
                            alpha=0.2, color='C3', zorder=10)
        if scale == "log":
            ax.set_yscale('log')
            ax.set_xscale('log')
        return ax
