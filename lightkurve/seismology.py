"""Defines the asteroseismology module"""
from __future__ import division, print_function

import copy
import os
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy
from astropy import units as u
from astropy import constants as const
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel


from uncertainties import ufloat
from uncertainties import umath

from . import MPLSTYLE

from scipy.signal import find_peaks

from .utils import LightkurveWarning


log = logging.getLogger(__name__)

#__all__ = ['estimate_radius','estimate_mass','estimate_logg']

"""Global parameters for the sun"""
NUMAX_SOL = ufloat(3090, 30) # microhertz | Huber et al. 2011
DELTANU_SOL = ufloat(135.1, 0.1) # microhertz | Huber et al. 2011
TEFF_SOL = ufloat(5772., 0.8) # Kelvin    | Prsa et al. 2016
G_SOL = ((const.G * const.M_sun)/(const.R_sun)**2).to(u.cm/u.second**2) #cms^2

class SeismologyButler(object):
    '''Good day, I am the Seismology Butler, I am here to help clean up your seismic mess.
    '''

    def __init__(self, periodogram):
        self.periodogram = periodogram
        self.frequency = periodogram.frequency
        self.power = periodogram.power
        self._numax_result = None
        self._deltanu_result = None

#        super(LombScarglePeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('SeismologyButler(ID: {})'.format(self.periodogram.targetid))

    @staticmethod
    def from_lightcurve(lightcurve, **kwargs):
        warnings.warn("Building a SeismologyButler object directly from a light curve "
                      "using default periodogram parameters. For further tuneability, "
                      "create a periodogram object first, using `to_periodogram`.")
        return SeismologyButler(periodogram=lightcurve.to_periodogram())

    @staticmethod
    def from_periodogram(periodogram, **kwargs):
        return SeismologyButler(periodogram=periodogram, **kwargs)

    def plot(self, **kwargs):
        return self.periodogram.plot(**kwargs)


    def _get_fwhm(self, numax):
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

        Parameters:
        ----------
            numax : float
                The estimated position of the numax of the power spectrum. This
                is used to calculated the region autocorrelated with itself.

        Returns:
        --------
            fwhm: float
                The estimate full-width-half-maximum of the seismic mode envelope
        """
        #Calculate the index FWHM for a given numax
        if u.Quantity(self.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
            fwhm = 0.25 * numax
        else:
            fwhm = 0.66 * numax**0.88
        return fwhm


    def _autocorrelate(self, numax, window=25., frequency_spacing=None):
        """An autocorrelation function (ACF) for seismic mode envelopes.
        We autocorrelate a region with a width of `window` (in microhertz)
        around a central frequency `numax` (in microhertz). The window size is
        determined based on the location of the nyquist frequency when
        estimating numax, and based on the expected width of the mode envelope
        of the asteroseismic oscillations when calculating deltanu.

        Parameters:
        ----------
            numax : float
                The estimated position of the numax of the power spectrum. This
                is used to calculated the region autocorrelated with itself.

            window : int or float
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
            frequency_spacing = np.median(np.diff(self.frequency.value))

        spread = int(window/2/frequency_spacing)                           # Find the spread in indices
        x = int(numax / frequency_spacing)                                 # Find the index value of numax
        x0 = int((self.frequency[0].value/frequency_spacing))              # Transform in case the index isn't from 0
        xt = x - x0
        p_sel = self.power[xt-spread:xt+spread].value       # Make the window selection

        C = np.correlate(p_sel, p_sel, mode='full')[len(p_sel)-1:]     #Correlated the resulting SNR space with itself
        return C


    def plot_echelle(self, deltanu, numax=None,
                    minimum_frequency=None, maximum_frequency=None,
                    scale='linear',
                    cmap='Blues'):
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

        # Ensure input deltanu is in the correct units
        deltanu = u.Quantity(deltanu, self.frequency.unit).value

        fmin = self.frequency[0].value
        fmax = self.frequency[-1].value

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

            fmin = numax - 2*self._get_fwhm(numax)
            if fmin < 0.:
                fmin = 0.

            fmax = numax + 2*self._get_fwhm(numax)
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
        if fmax < self.frequency[-1].value - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(self.frequency.value))
        x0 = int(self.frequency.value[0] / fs)

        ff = self.frequency[int(fmin/fs)-x0:int(fmax/fs)-x0].value   #The the selected frequency range
        pp = self.power[int(fmin/fs)-x0:int(fmax/fs)-x0].value   #The selected power range

        n_rows = int((ff[-1]-ff[0])/deltanu)     #The number of stacks to use
        n_columns = int(deltanu/fs)               #The number of elements in each stack

        #Reshape the power into n_rowss of n_columnss
        ep = np.reshape(pp[:(n_rows*n_columns)],(n_rows,n_columns))

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


    def _estimate_numax_acf(self, numaxs=None, window=None, spacing=None):
        """
        Helper function to perform the numax estimation for both the
        `estimate_numax()` and `plot_numax_diagnostics()` functions.

        For details, see the `estimate_numax()` function.
        """

        # Calculate the window size

        #C: What is this doing? Why have these values been picked? This function is slow.
        if window is None:
            if u.Quantity(self.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
                window = u.Quantity(250., u.microhertz).to(self.frequency.unit).value
            else:
                window = u.Quantity(25., u.microhertz).to(self.frequency.unit).value

        # Calculate the spacing size
        if spacing is None:
            if u.Quantity(self.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
                spacing = u.Quantity(10., u.microhertz).to(self.frequency.unit).value
            else:
                spacing = u.Quantity(1., u.microhertz).to(self.frequency.unit).value

        # Run some checks on the inputs
        window = u.Quantity(window, self.frequency.unit).value
        spacing = u.Quantity(spacing, self.frequency.unit).value
        if numaxs is None:
            numaxs = np.arange(np.ceil(np.nanmin(self.frequency.value)) + window/2,
                        np.floor(np.nanmax(self.frequency.value)) - window/2,
                        spacing)
        numaxs = u.Quantity(numaxs, self.frequency.unit).value
        if not hasattr(numaxs, '__iter__'):
            numaxs = np.asarray([numaxs])

        fs = np.median(np.diff(self.frequency.value))
        for var, label in zip([np.asarray(window), np.asarray(spacing), numaxs], ['window', 'spacing', 'numaxs']):
            if (var < fs).any():
                raise ValueError("You can't have {} smaller than the "
                                "frequency separation!".format(label))
            if (var > (self.frequency[-1].value - self.frequency[0].value)).any():
                raise ValueError("You can't have {} wider than the entire "
                                "power spectrum!".format(label))
            if (var < 0).any():
                raise ValueError("Please pass an entirely positive {}.".format(label))

        #We want to find the numax which returns in the highest autocorrelation
        #power, rescaled based on filter width
        fs = np.median(np.diff(self.frequency.value))

        metric = np.zeros(len(numaxs))
        acf2d = np.zeros([int(window/2/fs)*2,len(numaxs)])
        for idx, numax in enumerate(numaxs):
            acf = self._autocorrelate(numax, window=window, frequency_spacing=fs)      #Return the acf at this numax
            acf2d[:,idx] = acf                                     #Store the 2D acf
            metric[idx] = (np.sum(np.abs(acf)) - 1 ) / len(acf)  #Store the max acf power normalised by the length

        # Smooth the data to find the peak
        # Previous smoothing could be completely wrong, it's based on the length of the array, not the frequency!!!
        # It needs to be based on the frequency differences in `numaxs`
        if len(numaxs) > 10:
            g = Gaussian1DKernel(stddev=100 * np.nanmedian(np.diff(numaxs)))
            metric_smooth = convolve(metric, g, boundary='extend')
        else:
            metric_smooth = metric
        best_numax = numaxs[np.argmax(metric_smooth)]     #The highest value of the metric corresponds to numax

        # This should be a dictionary...
        result = {'best_numax':u.Quantity(best_numax, self.frequency.unit),
                'numaxs':numaxs, 'acf2d':acf2d, 'window':window, 'metric':metric,
                'metric_smooth': metric_smooth}
        return result


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

        Parameters:
        -----------
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

        Returns:
        --------
        numax : float
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        """

        r = self._estimate_numax_acf(numaxs, window, numax_spacing)
        self._numax_result = r
        return r['best_numax']

    def plot_numax_diagnostics(self, numaxs=None, window=None, numax_spacing=None, return_metric=False):
        """ Returns three diagnostic plots and an estimated value for numax.

        [1] The SNRPeriodogram plotted with a red line indicating the estimated
        numax value.

        [2] An image showing the 2D autocorrelation. On the y-axis is the
        frequency lag of the autocorrelation window. The width of the window is
        equal to `window`, and the spacing between lags is equal to
        `numax_spacing`. On the x-axis is the central frequency at which the
        autocorrelation was calculated. In the z-axis is the unitless
        autocorrelation power. Shown in red is the estimated numax.

        [3] The Mean Collapsed Correlation (MCC, see Viani et al. 2019) against
        central frequency at which the MCC was calculated. Shown in red is the
        estimated numax. Shown in blue is the MCC convolved with a Gaussian
        smoothing kernel with a standard deviation of 1/5th the window size.

        For details on the numax estimation, see the `estimate_numax()` function.
        The calculation performed is identical

        Parameters:
        -----------
        numaxs : array-like
            An array of numaxs at which to evaluate the autocorrelation. If
            none is given, a sensible range will be chosen.

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

        return_metric : bool
            If True, returns the metric data shown in the lower diagnostic plot.

        Returns:
        --------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if self._numax_result is None:
            self.estimate_numax(numaxs, window, numax_spacing)
        result = self._numax_result

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots(3, sharex=True, figsize=(8.485, 12))
            self.plot(ax=ax[0])
#            ax[0].set_ylabel(r'SNR')
#            ax[0].set_title(r'SNR vs Frequency')
            ax[0].set_xlabel('')

            windowarray = np.linspace(0, result['window'], num=result['acf2d'].shape[1])
            extent = (result['numaxs'][0], result['numaxs'][-1], windowarray[0], windowarray[-1])
            figsize = [8.485, 4]
            a = figsize[1] / figsize[0]
            b = (extent[3] - extent[2]) / (extent[1] - extent[0])

            ax[1].imshow(result['acf2d'],cmap='Blues', aspect=a/b, origin='lower',extent=extent)
            ax[1].set_ylabel(r'Frequency lag [{}]'.format(self.frequency.unit.to_string('latex')))

            ax[2].plot(result['numaxs'], result['metric'])
            ax[2].plot(result['numaxs'], result['metric_smooth'])
            ax[2].set_xlabel("Frequency [{}]".format(self.frequency.unit.to_string('latex')))
            ax[2].set_ylabel(r'Correlation Metric')
            ax[0].axvline(result['best_numax'].value,c='r', linewidth=2,alpha=.4)
            ax[1].axvline(result['best_numax'].value,c='r', linewidth=2,alpha=.4)
            ax[2].axvline(result['best_numax'].value,c='r', linewidth=2,alpha=.4,
                label=r'{:.1f} {}'.format(result['best_numax'].value,
                                    self.frequency.unit.to_string('latex')))
            ax[2].legend()

        return ax

    def _estimate_deltanu_acf(self, numax):
        """
        Helper function to perform the deltanu estimation for both the
        `estimate_deltanu()` and `plot_deltanu_diagnostics()` functions.

        For details, see the `estimate_deltanu()` function.
        """

        # Run some checks on the passed in numaxs
        # Ensure input numax is in the correct units
        numax = u.Quantity(numax, self.frequency.unit)
        fs = np.median(np.diff(self.frequency.value))
        if numax.value < fs:
            raise ValueError("The input numax can not be lower than"
                            " a single frequency bin.")
        if numax.value > np.nanmax(self.frequency.value):
            raise ValueError("The input numax can not be higher than"
                            "the highest frequency value in the periodogram.")


        #Calcluate deltanu using the method by Stello et al. 2009
        #Make sure that this relation only ever happens in microhertz space
        deltanu_emp = u.Quantity((0.294 * u.Quantity(numax, u.microhertz).value ** 0.772)*u.microhertz,
                            self.frequency.unit).value

        window = 2*int(np.floor(self._get_fwhm(numax.value)))
        aacf = self._autocorrelate(numax = numax.value, window=window)
        acf = (np.abs(aacf**2)/np.abs(aacf[0]**2)) / (3/(2*len(aacf)))
        fs = np.median(np.diff(self.frequency.value))
        lags = np.linspace(0., len(acf)*fs, len(acf))

        #Select a 25% region region around the empirical deltanu
        sel = (lags > deltanu_emp - .25*deltanu_emp) & (lags < deltanu_emp + .25*deltanu_emp)

        #Run a peak finder on this region
        peaks, _ = find_peaks(acf[sel], distance=np.floor(deltanu_emp/2. / fs))

        #Select the peak closest to the empirical value
        best_deltanu = lags[sel][peaks][np.argmin(np.abs(lags[sel][peaks] - deltanu_emp))]

        result = {'best_deltanu':u.Quantity(best_deltanu, self.frequency.unit),
                'lags':lags, 'acf':acf, 'peaks':peaks, 'sel':sel}
        return result

    def estimate_deltanu(self, numax=None, method='acf'):
        """ Estimates the average value of the large frequency spacing, DeltaNu,
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
        deltanu : float
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """
        if numax is None:
            if self._numax_result is None:
                self.estimate_numax()
            numax = self._numax_result['best_numax']
        r= self._estimate_deltanu_acf(numax)
        self._deltanu_result = r
        return r['best_deltanu']

    def plot_deltanu_diagnostics(self, numax=None, return_metric=False):
        """Returns one diagnostic plots and an estimated value for deltanu.

        [1] Scaled correlation metric vs frequecy lag of the autocorrelation
        window, with inset close up on the determined deltanu and a line
        indicating the determined deltanu.

        For details on the deltanu estimation, see the `estimate_deltanu()`
        function. The calculation performed is identical.

        NOTE: When plotting , we exclude the first frequency lag bin, to
        make the relevant features on the plot clearer.

        Parameters:
        -----------
        numax : float
            An estimated numax value of the mode envelope in the periodogram. If
            not given units it is assumed to be in units of the periodogram
            frequency attribute.

        return_metric : bool
            If True, returns the metric data shown in the lower diagnostic plot.

        Returns:
        --------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if self._deltanu_result is None:
            self.estimate_deltanu(numax)

        result = self._deltanu_result

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots()
            # ax.plot(lags, acf/acf[0])
            ax.plot(result['lags'][1:], result['acf'][1:])
            ax.set_xlabel("Frequency Lag [{}]".format(self.frequency.unit.to_string('latex')))
            ax.set_ylabel(r'Scaled Correlation')
            ax.axvline(result['best_deltanu'].value,c='r', linewidth=2,alpha=.4)
            ax.set_title(r'Scaled Correlation vs Lag for a given $\nu_{\rm max}$')

            axin = inset_axes(ax, width="50%",height="50%", loc="upper right")
            axin.set_yticks([])
            axin.plot(result['lags'][result['sel']],result['acf'][result['sel']])
            axin.scatter(result['lags'][result['sel']][result['peaks']], result['acf'][result['sel']][result['peaks']],c='r',s=5)
            axin.axvline(result['best_deltanu'].value,c='r', linewidth=2,alpha=.4,
                label=r'{:.1f} {}'.format(result['best_deltanu'].value,
                                    self.frequency.unit.to_string('latex')))
            axin.legend(loc='best')

            return ax


    def estimate_radius(numax, deltanu, Teff, numax_err=None, deltanu_err=None, Teff_err=None):
        """Calculates radius using the asteroseismic scaling relations.
        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar radius as

        R = Rsol * (numax/numax_sol)(deltanu/deltanusol)^-2(Teff/Teffsol)^0.5

        where R is the radius and Teff is the effective temperature, and the suffix
        'sol' indicates a solar value. In this method we use the solar values for
        numax and deltanu as given in Huber et al. (2011) and for Teff as given in
        Prsa et al. (2016).

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.

        deltanu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. If not given an astropy unit, assumed to be in units of
            microhertz.

        Teff : float
            The effective temperature of the star. In units of Kelvin.

        numax_err : float
            Error on numax. Assumed to be same units as numax

        deltanu_err : float
            Error on deltanu. Assumed to be same units as deltanu

        Teff_err : float
            Error on Teff. Assumed to be same units as Teff.

        Returns
        -------
        R : float
            An estimate of the stellar radius in solar radii.

        R_e : float
            Uncertainty on the stellar radius estimate in solar radii. Returned only
            if all of `numax_err`, `deltanu_err` and `teff_err` are passed.
        """
        numax = u.Quantity(numax, u.microhertz).value
        deltanu = u.Quantity(deltanu, u.microhertz).value
        Teff = u.Quantity(Teff, u. Kelvin).value

        if all(b is not None for b in [numax_err, deltanu_err, Teff_err]):
            numax_err = u.Quantity(numax_err, u.microhertz).value
            deltanu_err = u.Quantity(deltanu_err, u.microhertz).value
            Teff_err = u.Quantity(Teff_err, u.Kelvin).value
            unumax = ufloat(numax, numax_err)
            udeltanu = ufloat(deltanu, deltanu_err)
            uTeff = ufloat(Teff, Teff_err)

        else:
            unumax = ufloat(numax, 0)
            udeltanu = ufloat(deltanu, 0)
            uTeff = ufloat(Teff, 0)

        uR = (unumax / NUMAX_SOL) * (udeltanu / DELTANU_SOL)**(-2.) * (uTeff / TEFF_SOL)**(0.5)

        R = uR.n * u.solRad
        R_e = uR.s * u.solRad

        if all(b is not None for b in [numax_err, deltanu_err, Teff_err]):
            return R, R_e
        else:
            return R

    def estimate_mass(numax, deltanu, Teff, numax_err=None, deltanu_err=None, Teff_err=None):
        """Calculates mass using the asteroseismic scaling relations.
        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar mass as

        M = Msol * (numax/numax_sol)^3(deltanu/deltanusol)^-4(Teff/Teffsol)^1.5

        where M is the mass and Teff is the effective temperature, and the suffix
        'sol' indicates a solar value. In this method we use the solar values for
        numax and deltanu as given in Huber et al. (2011) and for Teff as given in
        Prsa et al. (2016).

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.

        deltanu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. If not given an astropy unit, assumed to be in units of
            microhertz.

        Teff : float
            The effective temperature of the star. In units of Kelvin.

        numax_err : float
            Error on numax. Assumed to be same units as numax

        deltanu_err : float
            Error on deltanu. Assumed to be same units as deltanu

        Teff_err : float
            Error on Teff. Assumed to be same units as Teff.

        Returns
        -------
        M : float
            An estimate of the stellar mass in solar masses.

        M_e : float
            Uncertainty on the stellar mass estimate in solar masses. Returned only
            if all of `numax_err`, `deltanu_err` and `teff_err` are passed.
        """
        numax = u.Quantity(numax, u.microhertz).value
        deltanu = u.Quantity(deltanu, u.microhertz).value
        Teff = u.Quantity(Teff, u.Kelvin).value

        if all(b is not None for b in [numax_err, deltanu_err, Teff_err]):
            numax_err = u.Quantity(numax_err, u.microhertz).value
            deltanu_err = u.Quantity(deltanu_err, u.microhertz).value
            Teff_err = u.Quantity(Teff_err, u.Kelvin).value

            unumax = ufloat(numax, numax_err)
            udeltanu = ufloat(deltanu, deltanu_err)
            uTeff = ufloat(Teff, Teff_err)

        else:
            unumax = ufloat(numax, 0)
            udeltanu = ufloat(deltanu, 0)
            uTeff = ufloat(Teff, 0)

        uM = (unumax / NUMAX_SOL)**3. * (udeltanu / DELTANU_SOL)**(-4.) * (uTeff / TEFF_SOL)**(1.5)

        M = uM.n * u.solMass
        M_e = uM.s * u.solMass

        if all(b is not None for b in [numax_err, deltanu_err, Teff_err]):
            return M, M_e
        else:
            return M

    def estimate_logg(numax, Teff, numax_err=None, Teff_err=None):
        """Calculates the log of the surface gravity using the asteroseismic scaling
        relations.
        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar surface gravity as

        g = gsol * (numax/numax_sol)(Teff/Teffsol)^0.5

        where g is the surface gravity and Teff is the effective temperature,
        and the suffix 'sol' indicates a solar value. In this method we use the
        solar values for numax as given in Huber et al. (2011) and for Teff as given
        in Prsa et al. (2016). The solar surface gravity is calcluated from the
        astropy constants for solar mass and radius and does not have an error.

        The solar surface gravity is returned as log10(g) with units in dex, as is
        common in the astrophysics literature.

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.

        Teff : float
            The effective temperature of the star. In units of Kelvin.

        numax_err : float
            Error on numax. Assumed to be same units as numax

        Teff_err : float
            Error on Teff. Assumed to be same units as Teff.

        Returns
        -------
        logg : float
            The log10 of the surface gravity of the star.

        logg_e : float
            Uncertainty on the log10 of the surface gravity in dex. Returned only
            if both of `numax_err` and `teff_err` are passed.
        """
        numax = u.Quantity(numax, u.microhertz).value
        Teff = u.Quantity(Teff, u.Kelvin).value

        if all(b is not None for b in [numax_err, Teff_err]):
            numax_err = u.Quantity(numax_err, u.microhertz).value
            Teff_err = u.Quantity(Teff_err, u.Kelvin).value

            unumax = ufloat(numax, numax_err)
            uTeff = ufloat(Teff, Teff_err)

        else:
            unumax = ufloat(numax, 0)
            uTeff = ufloat(Teff, 0)

        ug = G_SOL.value * (unumax / NUMAX_SOL) * (uTeff / TEFF_SOL)**0.5
        ulogg = umath.log(ug, 10)

        logg = ulogg.n * u.dex
        logg_e = ulogg.s * u.dex

        if all(b is not None for b in [numax_err, Teff_err]):
            return  logg, logg_e
        return logg
