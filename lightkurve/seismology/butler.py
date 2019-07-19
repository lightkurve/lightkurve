"""Defines the SeismologyButler class.

TODO
----
* Consider putting Teff in repr of stellar parameters.
* Errors are not yet passed to stellar_estimator functions.
* Clean up docstrings and plots.
"""
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.units import cds

from .. import MPLSTYLE
from . import utils, stellar_estimators
from ..periodogram import LombScarglePeriodogram, SNRPeriodogram
from ..utils import LightkurveWarning

log = logging.getLogger(__name__)

__all__ = ['SeismologyButler']


class SeismologyButler(object):
    """Good day, I am the Seismology Butler. I am here to help you do asteroseismology.

    This class provides easy access to methods to estimate numax and deltanu,
    and stores them on its tray for easy diagnostic plotting.
    """
    def __init__(self, periodogram):
        if not isinstance(periodogram, SNRPeriodogram):
            warnings.warn("SeismologyButler received a periodogram which does not "
                          "appear to have been background-corrected. Consider calling "
                          "`periodogram.flatten()` prior to extracting seismological parameters.",
                          LightkurveWarning)
        self.periodogram = periodogram

    def __repr__(self):
        attrs = np.asarray(['numax', 'deltanu', 'mass', 'radius', 'logg'])
        tray = np.asarray([hasattr(self, attr) for attr in attrs])
        if tray.sum() == 0:
            tray_str = '\n\t|  Tray is empty.  |\n\t ' + '‾'*(18 + len(', '.join(attrs[tray])))
        else:
            tray_str = '\n\t|  On tray: ' + ', '.join(attrs[tray])+'  |\n\t '+'‾'*(13 + len(', '.join(attrs[tray])))


        return 'SeismologyButler(ID: {})\n{}'.format(self.periodogram.targetid, tray_str)

    @staticmethod
    def from_lightcurve(lc, **kwargs):
        """Returns a `SeismologyButler` given a `LightCurve` object."""
        log.info("Building a SeismologyButler object directly from a light curve "
                 "uses default periodogram parameters. For further tuneability, "
                 "create a periodogram object first, using `to_periodogram`.")
        return SeismologyButler(periodogram=lc.to_periodogram(**kwargs).flatten())

    def _validate_method(self, method, supported_methods):
        """Raises ValueError if a method is not supported."""
        if method in supported_methods:
            return method
        raise ValueError("method {} is not supported; "
                         "must be one of {}".format(method, supported_methods))

    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `SeismologyButler"
                                     ".estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError("You need to call `SeismologyButler"
                                     ".estimate_deltanu()` first.")
        return deltanu

    def plot_echelle(self, deltanu=None, numax=None,
                     minimum_frequency=None, maximum_frequency=None,
                     scale='linear', cmap='Blues'):
        """Plots an echelle diagram of the periodogram by stacking the
        periodogram in slices of deltanu. Modes of equal radial degree should
        appear approximately vertically aligned. If no structure is present,
        you are likely dealing with a faulty deltanu value or a low signal to noise
        case.

        This method is adapted from work by Daniel Hey & Guy Davies.

        Parameters
        ----------
        deltanu : float
            Value for the large frequency separation of the seismic mode
            frequencies in the periodogram. Assumed to have the same units as
            the frequencies, unless given an Astropy unit.
            Is assumed to be in the same units as frequency if not given a unit.
        numax : float
            Value for the frequency of maximum oscillation. If a numax is
            passed, a suitable range one FWHM of the mode envelope either side
            of the will be shown. This is overwritten by custom frequency ranges.
            Is assumed to be in the same units as frequency if not given a unit.
        minimum_frequency : float
            The minimum frequency at which to display the echelle
            Is assumed to be in the same units as frequency if not given a unit.
        maximum_frequency : float
            The maximum frequency at which to display the echelle.
            Is assumed to be in the same units as frequency if not given a unit.
        scale: str
            Set z axis to be "linear" or "log". Default is linear.
        cmap : str
            The name of the matplotlib colourmap to use in the echelle diagram.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        freq = self.periodogram.frequency  # Makes code below more readable

        fmin = freq[0]
        fmax = freq[-1]

        # Check for any superfluous input
        if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            warnings.warn("You have passed both a numax and a frequency limit. "
                          "The frequency limit will override the numax input.",
                          LightkurveWarning)

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, freq.unit).value
            if numax > freq[-1].value:
                raise ValueError("You can't pass in a numax outside the"
                                "frequency range of the periodogram.")

            fwhm = utils.get_fwhm(self.periodogram, numax)

            fmin = numax - 2*fwhm
            if fmin < 0.:
                fmin = 0.

            fmax = numax + 2*fwhm
            if fmax > freq[-1].value:
                fmax = freq[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin =  u.Quantity(minimum_frequency, freq.unit).value
            if fmin > freq[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                 "frequency range of the periodogram.")

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, freq.unit).value
            if fmax > freq[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                 "frequency range of the periodogram.")

        # Make sure fmin and fmax are Quantities or code below will break
        fmin = u.Quantity(fmin, freq.unit)
        fmax = u.Quantity(fmax, freq.unit)

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < freq[-1] - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(freq))
        x0 = int(freq[0] / fs)

        ff = freq[int(fmin/fs)-x0:int(fmax/fs)-x0] # Selected frequency range
        pp = self.periodogram.power[int(fmin/fs)-x0:int(fmax/fs)-x0] # Power range

        n_rows = int((ff[-1]-ff[0])/deltanu) # Number of stacks to use
        n_columns = int(deltanu/fs)          # Number of elements in each stack

        # Reshape the power into n_rows of n_columns
        ep = np.reshape(pp[:(n_rows*n_columns)], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        #Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[:(n_rows*n_columns)],(n_rows,n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        y_f = (ef[:,0])

        #Plot the echelle diagram
        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots()
            extent = (x_f[0].value, x_f[-1].value, y_f[0].value, y_f[-1].value)
            figsize = plt.rcParams['figure.figsize']
            a = figsize[1] / figsize[0]
            b = (extent[3] - extent[2]) / extent[1]
            vmin = np.nanpercentile(ep.value, 1)
            vmax = np.nanpercentile(ep.value, 99)
            im = ax.imshow(ep.value, cmap=cmap, aspect=a/b, origin='lower',
                      extent=extent, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax)
            if isinstance(self.periodogram, SNRPeriodogram):
                ylabel = 'Signal to Noise Ratio (SNR)'
            elif self.periodogram.power.unit == cds.ppm:
                ylabel = "Amplitude [{}]".format(self.periodogram.power.unit.to_string('latex'))
            else:
                ylabel = "Power Spectral Density [{}]".format(self.periodogram.power.unit.to_string('latex'))

            cbar.set_label(ylabel)
            ax.set_xlabel(r'Frequency mod. {:.2f}'.format(deltanu))
            ax.set_ylabel(r'Frequency [{}]'.format(freq.unit.to_string('latex')))
            ax.set_title('Echelle diagram for {}'.format(self.periodogram.label))

        return ax


    def estimate_numax(self, method="acf", numaxs=None, window_width=None, spacing=None):
        """Estimates the peak of the envelope of seismic oscillation modes,
        numax using an autocorrelation function. There are many papers on the
        topic of autocorrelation functions for estimating seismic parameters,
        including but not limited to: Roxburgh & Vorontsov (2006),
        Roxburgh (2009), Mosser & Appourchaux (2009), Huber et al. (2009),
        Verner & Roxburgh (2011) & Viani et al. (2019).

        We base this approach first and foremost off the 2D ACF numax estimation
        presented in Viani et al. (2019) and other papers above. A window of
        fixed width (either given by the user, 25 microhertz for Red Giants or
        250 microhertz for Main Sequence stars) is moved along the power
        spectrum, where the central frequency of the window moves in steps of 1
        microhertz (or given by the user as `spacing`) and evaluates the
        autocorrelation at each step.

        The correlation (numpy.correlate) is typically given as:

        C[x, y] = sum( x * conj(y) ) .

        The autocorrelation power of a full spectrum with itself is then

        C = sum(s * s),

        where s is a window of the signal-to-noise spectrum.
        In order to evaluate where the correlation power is highest (indicative
        of the power excess of the modes) we calculate the Mean Collapsed
        Correlation (MCC, see Kiefer 2013, Viani et al. 2019) as

        MCC = (sum(|C|) - 1) / nlags ,

        where C is the autocorrelation power at a given central freqeuncy, and
        nlags is the number of lags in the autocorrelation.

        The MCC metric is covolved with an Astropy Gaussian 1D Kernel with a
        standard deviation of 1/5th of the window size to smooth it. The
        frequency that results in the highest value of the smoothed MCC is the
        detected numax.

        NOTE: This method is not robust against large peaks in the spectrum (due
        to e.g. spacecraft rotation), nor is it robust in the case of low signal
        to noise (such as for single sector TESS data). Exercise caution when
        using this module!

        NOTE: This function is intended for use with solar like Main Sequence
        and Red Giant Branch oscillators only.

        Parameters
        ----------
        numaxs : array-like
            An array of numaxs at which to evaluate the autocorrelation. If
            none is given, a sensible range will be chosen. If no units are
            given it is assumed to be in the same units as the periodogram
            frequency.
        window_width : int or float
            The width of the autocorrelation window around each central
            frequency in 'numaxs'. If none is given, a sensible value will be
            chosen. If no units are given it is assumed to be in the same units
            as the periodogram frequency.
        spacing : int or float
            The spacing between central frequencies (numaxs) at which the
            autocorrelation is evaluated. If none is given, a sensible value
            will be assumed. If no units are given it is assumed to be in the
            same units as the periodogram frequency.

        Returns
        -------
        numax : `SeismologyQuantity`
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        """
        method = self._validate_method(method, supported_methods=["acf"])
        if method == "acf":
            from .numax_estimators import estimate_numax_acf
            result = estimate_numax_acf(self.periodogram, numaxs=numaxs,
                                        window_width=window_width, spacing=spacing)
        self.numax = result
        return result

    def diagnose_numax(self, numax=None):
        """Create diagnostic plots showing how numax was estimated."""
        numax = self._validate_numax(numax)
        return numax.diagnostics_plot_method(numax, self.periodogram)

    def estimate_deltanu(self, method='acf', numax=None):
        """Estimates the average value of the large frequency spacing, DeltaNu,
        of the seismic oscillations of the target, using an autocorrelation
        function. There are many papers on the topic of autocorrelation
        functions for estimating seismic parameters, including but not limited
        to: Roxburgh & Vorontsov (2006), Roxburgh (2009),
        Mosser & Appourchaux (2009), Huber et al. (2009),
        Verner & Roxburgh (2011) & Viani et al. (2019).

        We base this approach first and foremost off the approach taken in
        Mosser & Appourchaux (2009). Given a known numax, a window around this
        numax is taken of one estimated full-width-half-maximum (FWHM) of the
        seismic mode envelope either side of numax. This width is chosen so that
        the autocorrelation includes all of the visible mode peaks.

        The autocorrelation (numpy.correlate) is given as:

        C = sum(s * s)

        where s is a window of the signal-to-noise spectrum. When shifting
        the spectrum over itself, C will increase when two mode peaks are
        overlapping.

        As is done in Mosser & Appourchaux, we rescale the value of C in terms
        of the noise level in the ACF spectrum as

        A = |C^2| / |C[0]^2|) * (2 * len(C) / 3) .

        The method will autocorrelate the region around the estimated numax
        expected to contain seismic oscillation modes. Repeating peaks in the
        autocorrelation implies an evenly spaced structure of modes.
        The peak closest to an empirical estimate of deltanu is taken as the true
        value. The peak finding algorithm is limited by a minimum spacing
        between peaks of 0.5 times the empirical value for deltanu.

        Our empirical estimate for numax is taken from Stello et al. (2009) as

        deltanu = 0.294 * numax^0.772

        If `numax` is None, a numax is calculated using the estimate_numax()
        function with default settings.

        NOTE: This function is intended for use with solar like Main Sequence
        and Red Giant Branch oscillators only.

        Parameters:
        ----------
        numax : float
            An estimated numax value of the mode envelope in the periodogram. If
            not given units it is assumed to be in units of the periodogram
            frequency attribute.

        Returns:
        -------
        deltanu : `SeismologyQuantity`
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """
        method = self._validate_method(method, supported_methods=["acf"])
        numax = self._validate_numax(numax)

        if method == "acf":
            from .deltanu_estimators import estimate_deltanu_acf
            result = estimate_deltanu_acf(self.periodogram, numax=numax)

        self.deltanu = result
        return result

    def diagnose_deltanu(self, method="acf", deltanu=None):
        """Create diagnostic plots showing how numax was estimated."""
        deltanu = self._validate_deltanu(deltanu)
        return deltanu.diagnostics_plot_method(deltanu, self.periodogram)

    def estimate_radius(self, teff, numax=None, deltanu=None):
        """Returns a stellar radius estimate based on the scaling relations."""
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        result = stellar_estimators.estimate_radius(numax, deltanu, teff)
        self.radius = result
        return result

    def estimate_mass(self, teff, numax=None, deltanu=None):
        """Returns a stellar mass estimate based on the scaling relations."""
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        result = stellar_estimators.estimate_mass(numax, deltanu, teff)
        self.mass = result
        return result

    def estimate_logg(self, teff, numax=None):
        """Returns a surface gravity estimate based on the scaling relations."""
        numax = self._validate_numax(numax)
        result = stellar_estimators.estimate_logg(numax, teff)
        self.logg = result
        return result
