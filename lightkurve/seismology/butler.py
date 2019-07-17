"""Defines the SeismologyButler class.

TODO
----
* Considering putting Teff in repr of stellar parameters.
* Errors are not yet passed to stellar_estimator functions.
* `plot_echelle` is broken due to a Quantity problem; should be easy to fix.
* Clean up docstrings.
* Clean up plots.
"""
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u

from .. import MPLSTYLE
from . import numax_estimators, deltanu_estimators, stellar_estimators
from . import utils

log = logging.getLogger(__name__)

__all__ = ['SeismologyButler']


class SeismologyButler(object):
    """Good day, I am the Seismology Butler, I am here to help clean up your seismic mess.
    
    This class provides easy access to methods to estimate numax and deltanu,
    and stores them on its tray for easy diagnostic plotting.
    """
    def __init__(self, periodogram):
        self.periodogram = periodogram
        self.frequency = periodogram.frequency
        self.power = periodogram.power
        self.numax = None
        self.deltanu = None
        self.radius = None

    def __repr__(self):
        return 'SeismologyButler(ID: {})'.format(self.periodogram.targetid)

    @staticmethod
    def from_lightcurve(lightcurve, **kwargs):
        warnings.warn("Building a SeismologyButler object directly from a light curve "
                      "using default periodogram parameters. For further tuneability, "
                      "create a periodogram object first, using `to_periodogram`.")
        return SeismologyButler(periodogram=lightcurve.to_periodogram())

    def _validate_numax(self, numax):
        """Raises an exception if both `numax` and `self.numax` are `None`."""
        if numax is None:
            if self.numax is None:
                raise ValueError("You need to call `butler.estimate_numax()` first.")
            return self.numax
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises an exception if both `deltanu` and `self.deltanu` are `None`."""
        if deltanu is None:
            if self.deltanu is None:
                raise ValueError("You need to call `butler.estimate_deltanu()` first.")
            return self.deltanu
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
        deltanu = self._validate_deltanu(deltanu)

        fmin = self.frequency[0]
        fmax = self.frequency[-1]

        # Check for any superfluous input
        if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            warnings.warn('You have passed both a numax and a frequency limit'
                          'The frequency limit will override the numax input')

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, self.frequency.unit).value
            if numax > self.frequency[-1].value:
                raise ValueError("You can't pass in a numax outside the"
                                "frequency range of the periodogram.")

            fmin = numax - 2*utils.get_fwhm(numax)
            if fmin < 0.:
                fmin = 0.

            fmax = numax + 2*utils.get_fwhm(numax)
            if fmax > self.frequency[-1].value:
                fmax = self.frequency[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin =  u.Quantity(minimum_frequency, self.frequency.unit).value
            if fmin > self.frequency[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                "frequency range of the periodogram.")

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, self.frequency.unit).value
            if fmax > self.frequency[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                "frequency range of the periodogram.")

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < self.frequency[-1] - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(self.frequency))
        x0 = int(self.frequency[0] / fs)

        ff = self.frequency[int(fmin/fs)-x0:int(fmax/fs)-x0]   # The the selected frequency range
        pp = self.power[int(fmin/fs)-x0:int(fmax/fs)-x0]   # The selected power range

        n_rows = int((ff[-1]-ff[0])/deltanu)     # The number of stacks to use
        n_columns = int(deltanu/fs)               # The number of elements in each stack

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

            extent = (x_f[0],x_f[-1],y_f[0],y_f[-1])
            figsize = plt.rcParams['figure.figsize']
            a = figsize[1]/figsize[0]
            b = (extent[3]-extent[2])/extent[1]

            ax.imshow(ep,cmap=cmap, aspect=a/b, origin='lower',
                     extent=extent)

            ax.set_xlabel(r'Frequency mod. {:.2f} {}'.format(deltanu,
                                        self.frequency.unit.to_string('latex')))
            ax.set_ylabel(r'Frequency [{}]'.format(self.frequency.unit.to_string('latex')))
            ax.set_title('Echelle diagram for {}'.format(self.label))

        return ax


    def estimate_numax(self, numaxs=None, window=None, numax_spacing=None):
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
        microhertz (or given by the user as `numax_spacing`) and evaluates the
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
        window : int or float
            The width of the autocorrelation window around each central
            frequency in 'numaxs'. If none is given, a sensible value will be
            chosen. If no units are given it is assumed to be in the same units
            as the periodogram frequency.
        numax_spacing : int or float
            The spacing between central frequencies (numaxs) at which the
            autocorrelation is evaluated. If none is given, a sensible value
            will be assumed. If no units are given it is assumed to be in the
            same units as the periodogram frequency.

        Returns
        -------
        numax : SeismologyResult
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        """
        result = numax_estimators.estimate_numax_acf(self.periodogram,
                                                     numaxs,
                                                     window,
                                                     numax_spacing)
        self.numax = result
        return result

    def diagnose_numax(self, numax=None):
        """Create diagnostic plots showing how numax was estimated."""
        numax = self._validate_numax(numax)
        return numax.diagnostics_plot_method(numax, self.periodogram)

    def estimate_deltanu(self, numax=None, method='acf'):
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
        deltanu : SeismologyResult
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """
        numax = self._validate_numax(numax)
        result = deltanu_estimators.estimate_deltanu_acf(self.periodogram,
                                                         numax=numax)
        self.deltanu = result
        return result

    def diagnose_deltanu(self, deltanu=None):
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
        """Returns a surface gravty estimate based on the scaling relations."""
        numax = self._validate_numax(numax)
        result = stellar_estimators.estimate_logg(numax, teff)
        self.logg = result
        return result
