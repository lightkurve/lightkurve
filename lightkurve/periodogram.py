"""Defines the Periodogram class and associated tools."""
from __future__ import division, print_function

import copy
import logging
import math
import warnings

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import astropy
from astropy.table import Table
from astropy import units as u
from astropy.units import cds
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel

# LombScargle was moved from astropy.stats to astropy.timeseries in AstroPy v3.2
try:
    from astropy.timeseries import LombScargle
except ImportError:
    from astropy.stats import LombScargle

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from . import MPLSTYLE

from .utils import LightkurveWarning

log = logging.getLogger(__name__)

__all__ = ['Periodogram', 'LombScarglePeriodogram', 'BoxLeastSquaresPeriodogram']


class Periodogram(object):
    """Generic class to represent a power spectrum (frequency vs power data).

    The Periodogram class represents a power spectrum, with values of
    frequency on the x-axis (in any frequency units) and values of power on the
    y-axis (in units of ppm^2 / [frequency units]).

    Attributes
    ----------
    frequency : `astropy.units.Quantity` object
        Array of frequencies with associated astropy unit.
    power : `astropy.units.Quantity` object
        Array of power-spectral-densities. The Quantity array must have units
        of `ppm^2 / freq_unit`, where freq_unit is the unit of the frequency
        attribute.
    nyquist : float, optional
        The Nyquist frequency of the lightcurve. In units of freq_unit, where
        freq_unit is the unit of the frequency attribute.
    label : str, optional
        Human-friendly object label, e.g. "KIC 123456789".
    targetid : str, optional
        Identifier of the target.
    default_view : "frequency" or "period"
        Should plots be shown in frequency space or period space by default?
    meta : dict, optional
        Free-form metadata associated with the Periodogram.
    """
    def __init__(self, frequency, power, nyquist=None, label=None,
                 targetid=None, default_view='frequency', meta={}):
        # Input validation
        if not isinstance(frequency, u.quantity.Quantity):
            raise ValueError('frequency must be an `astropy.units.Quantity` object.')
        if not isinstance(power, u.quantity.Quantity):
            raise ValueError('power must be an `astropy.units.Quantity` object.')
        # Frequency must have frequency units
        try:
            frequency.to(u.Hz)
        except u.UnitConversionError:
            raise ValueError('Frequency must be in units of 1/time.')
        # Frequency and power must have sensible shapes
        if frequency.shape[0] <= 1:
            raise ValueError('frequency and power must have a length greater than 1.')
        if frequency.shape != power.shape:
            raise ValueError('frequency and power must have the same length.')

        self.frequency = frequency
        self.power = power
        self.nyquist = nyquist
        self.label = label
        self.targetid = targetid
        self.default_view = self._validate_view(default_view)
        self.meta = meta

    def _validate_view(self, view):
        """Verifies whether `view` is is one of {"frequency", "period"} and
        raises a helpful `ValueError` if not.
        """
        if view is None and hasattr(self, 'default_view'):
            view = self.default_view
        allowed_views = ["frequency", "period"]
        if view not in allowed_views:
            raise ValueError(("'{}' is an invalid value for view, "
                              "allowed values are: {}.")
                             .format(view, allowed_views))
        return view

    @property
    def period(self):
        """Returns the array of periods, i.e. 1/frequency."""
        return 1. / self.frequency

    @property
    def max_power(self):
        """Returns the power of the highest peak in the periodogram."""
        return np.nanmax(self.power)

    @property
    def frequency_at_max_power(self):
        """Returns the frequency corresponding to the highest peak in the periodogram."""
        return self.frequency[np.nanargmax(self.power)]

    @property
    def period_at_max_power(self):
        """Returns the period corresponding to the highest peak in the periodogram."""
        return 1. / self.frequency_at_max_power

    def bin(self, binsize=10, method='mean'):
        """Bins the power spectrum.

        Parameters
        ----------
        binsize : int
            The factor by which to bin the power spectrum, in the sense that
            the power spectrum will be smoothed by taking the mean in bins
            of size N / binsize, where N is the length of the original
            frequency array. Defaults to 10.
        method : str, one of 'mean' or 'median'
            Method to use for binning. Default is 'mean'.

        Returns
        -------
        binned_periodogram : a `Periodogram` object
            Returns a new `Periodogram` object which has been binned.
        """
        # Input validation
        if binsize < 1:
            raise ValueError('binsize must be larger than or equal to 1')
        if method not in ('mean', 'median'):
            raise ValueError("{} is not a valid method, must be 'mean' or 'median'.".format(method))

        m = int(len(self.power) / binsize)  # length of the binned arrays
        if method == 'mean':
            binned_freq = self.frequency[:m*binsize].reshape((m, binsize)).mean(1)
            binned_power = self.power[:m*binsize].reshape((m, binsize)).mean(1)
        elif method == 'median':
            binned_freq = np.nanmedian(self.frequency[:m*binsize].reshape((m, binsize)), axis=1)
            binned_power = np.nanmedian(self.power[:m*binsize].reshape((m, binsize)), axis=1)

        binned_pg = self.copy()
        binned_pg.frequency = binned_freq
        binned_pg.power = binned_power
        return binned_pg

    def smooth(self, method='boxkernel', filter_width=0.1):
        """Smooths the power spectrum using the 'boxkernel' or 'logmedian' method.

        If `method` is set to 'boxkernel', this method will smooth the power
        spectrum by convolving with a numpy Box1DKernel with a width of
        `filter_width`, where `filter width` is in units of frequency.
        This is best for filtering out noise while maintaining seismic mode
        peaks. This method requires the Periodogram to have an evenly spaced
        grid of frequencies. A `ValueError` exception will be raised if this is
        not the case.

        If `method` is set to 'logmedian', it smooths the power spectrum using
        a moving median which moves across the power spectrum in a steps of

        log10(x0) + 0.5 * filter_width

        where `filter width` is in log10(frequency) space. This is best for
        estimating the noise background, as it filters over the seismic peaks.

        Periodograms that are unsmoothed have multiplicative noise that is
        distributed as chi squared 2 degrees of freedom.  This noise
        distribution has a well defined mean and median but the two are not
        equivalent.  The mean of a chi squared 2 dof distribution is 2, but the
        median is 2(8/9)**3.
        (see https://en.wikipedia.org/wiki/Chi-squared_distribution)
        In order to maintain consistency between 'boxkernel' and 'logmedian' a
        correction factor of (8/9)**3 is applied to (i.e., the median is divided
        by the factor) to the median values.

        In addition to consistency with the 'boxkernel' method, the correction
        of the median values is useful when applying the periodogram flatten
        method.  The flatten method divides the periodgram by the smoothed
        periodogram using the 'logmedian' method.  By appyling the correction
        factor we follow asteroseismic convention that the signal-to-noise
        power has a mean value of unity.  (note the signal-to-noise power is
        really the signal plus noise divided by the noise and hence should be
        unity in the absence of any signal)

        Parameters
        ----------
        method : str, one of 'boxkernel' or 'logmedian'
            The smoothing method to use. Defaults to 'boxkernel'.
        filter_width : float
            If `method` = 'boxkernel', this is the width of the smoothing filter
            in units of frequency.
            If method = `logmedian`, this is the width of the smoothing filter
            in log10(frequency) space.

        Returns
        -------
        smoothed_pg : `Periodogram` object
            Returns a new `Periodogram` object in which the power spectrum
            has been smoothed.
        """
        # Input validation
        if method not in ('boxkernel', 'logmedian'):
            raise ValueError("the `method` parameter must be one of "
                             "'boxkernel' or 'logmedian'.")

        if method == 'boxkernel':
            if filter_width <= 0.:
                raise ValueError("the `filter_width` parameter must be "
                                 "larger than 0 for the 'boxkernel' method.")
            try:
                filter_width = u.Quantity(filter_width, self.frequency.unit)
            except u.UnitConversionError:
                raise ValueError("the `filter_width` parameter must have "
                                 "frequency units.")

            # Check to see if we have a grid of evenly spaced periods instead.
            fs = np.mean(np.diff(self.frequency))
            if not np.isclose(np.median(np.diff(self.frequency.value)), fs.value):
                raise ValueError("the 'boxkernel' method requires the periodogram "
                                 "to have a grid of evenly spaced frequencies.")

            box_kernel = Box1DKernel(math.ceil((filter_width/fs).value))
            smooth_power = convolve(self.power.value, box_kernel)
            smooth_pg = self.copy()
            smooth_pg.power = u.Quantity(smooth_power, self.power.unit)
            return smooth_pg

        if method == 'logmedian':
            if isinstance(filter_width, astropy.units.quantity.Quantity):
                raise ValueError("the 'logmedian' method requires a dimensionless "
                                 "value for `filter_width` in log10(frequency) space.")
            count = np.zeros(len(self.frequency.value), dtype=int)
            bkg = np.zeros_like(self.frequency.value)
            x0 = np.log10(self.frequency[0].value)
            corr_factor = (8.0 / 9.0)**3
            while x0 < np.log10(self.frequency[-1].value):
                m = np.abs(np.log10(self.frequency.value) - x0) < filter_width
                if len(bkg[m] > 0):
                    bkg[m] += np.nanmedian(self.power[m].value) / corr_factor
                    count[m] += 1
                x0 += 0.5 * filter_width
            bkg /= count
            smooth_pg = self.copy()
            smooth_pg.power = u.Quantity(bkg, self.power.unit)
            return smooth_pg

    def plot(self, scale='linear', ax=None, xlabel=None, ylabel=None, title='',
             style='lightkurve', view=None, unit=None, **kwargs):
        """Plots the Periodogram.

        Parameters
        ----------
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        view : str
            {'frequency', 'period'}. Default 'frequency'. If 'frequency', x-axis
            units will be frequency. If 'period', the x-axis units will be
            period and 'log' scale.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if isinstance(unit, u.quantity.Quantity):
            unit = unit.unit

        view = self._validate_view(view)

        if unit is None:
            unit = self.frequency.unit
            if view == 'period':
                unit = self.period.unit

        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        if ylabel is None:
            if self.power.unit == cds.ppm:
                ylabel = "Amplitude [{}]".format(self.power.unit.to_string('latex'))
            else:
                ylabel = "Power Spectral Density [{}]".format(self.power.unit.to_string('latex'))

        # This will need to be fixed with housekeeping. Self.label currently doesnt exist.
        if ('label' not in kwargs) and ('label' in dir(self)):
            kwargs['label'] = self.label

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot frequency and power
            if view == 'frequency':
                ax.plot(self.frequency.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Frequency [{}]".format(unit.to_string('latex'))
            elif view == 'period':
                ax.plot(self.period.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Period [{}]".format(unit.to_string('latex'))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend()
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.set_title(title)
        return ax

    def plot_echelle(self, dnu, numax=None,
                    minimum_frequency=None, maximum_frequency=None,
                    scale='linear',
                    cmap='Blues'):
        """Plots an echelle diagram of the periodogram by stacking the
        periodogram in slices of dnu. Modes of equal radial degree should
        appear approximately vertically aligned. If no structure is present,
        you are likely dealing with a faulty dnu value or a low signal to noise
        case.

        This method is adapted from work by Daniel Hey & Guy Davies.

        Parameters
        ----------
        dnu : float
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

        # Ensure input dnu is in the correct units
        dnu = u.Quantity(dnu, self.frequency.unit).value

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

        # Add on 1x Dnu so we don't miss off any important range due to rounding
        if fmax < self.frequency[-1].value - 1.5*dnu:
            fmax += dnu

        fs = np.median(np.diff(self.frequency.value))

        ff = self.frequency[int(fmin/fs):int(fmax/fs)].value   #The the selected frequency range
        pp = self.power[int(fmin/fs):int(fmax/fs)].value   #The selected power range

        n_rows = int((ff[-1]-ff[0])/dnu)     #The number of stacks to use
        n_columns = int(dnu/fs)               #The number of elements in each stack

        #Reshape the power into n_rowss of n_columnss
        ep = np.reshape(pp[:(n_rows*n_columns)],(n_rows,n_columns))

        if scale=='log':
            ep = np.log10(ep)

        #Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[:(n_rows*n_columns)],(n_rows,n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % dnu)
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

            ax.set_xlabel(r'Frequency mod. {:.2f} {}'.format(dnu,
                                        self.frequency.unit.to_string('latex')))
            ax.set_ylabel(r'Frequency [{}]'.format(self.frequency.unit.to_string('latex')))
            ax.set_title('Echelle diagram for {}'.format(self.label))

        return ax

    def flatten(self, method='logmedian', filter_width=0.01, return_trend=False):
        """Estimates the Signal-To-Noise (SNR) spectrum by dividing out an
        estimate of the noise background.

        This method divides the power spectrum by a background estimated
        using a moving filter in log10 space by default. For details on the
        `method` and `filter_width` parameters, see `Periodogram.smooth()`

        Dividing the power through by the noise background produces a spectrum
        with no units of power. Since the signal is divided through by a measure
        of the noise, we refer to this as a `Signal-To-Noise` spectrum.

        Parameters
        ----------
        method : str, one of 'boxkernel' or 'logmedian'
            Background estimation method passed on to `Periodogram.smooth()`.
            Defaults to 'logmedian'.
        filter_width : float
            If `method` = 'boxkernel', this is the width of the smoothing filter
            in units of frequency.
            If method = `logmedian`, this is the width of the smoothing filter
            in log10(frequency) space.
        return_trend : bool
            If True, then the background estimate, alongside the SNR spectrum,
            will be returned.

        Returns
        -------
        snr_spectrum : `Periodogram` object
            Returns a periodogram object where the power is an estimate of the
            signal-to-noise of the spectrum, creating by dividing the powers
            with a simple estimate of the noise background using a smoothing filter.
        bkg : `Periodogram` object
            The estimated power spectrum of the background noise. This is only
            returned if `return_trend = True`.
        """
        bkg = self.smooth(method=method, filter_width=filter_width)
        snr_pg = self / bkg.power
        snr = SNRPeriodogram(snr_pg.frequency, snr_pg.power,
                             nyquist=self.nyquist, targetid=self.targetid,
                             label=self.label, meta=self.meta)
        if return_trend:
            return snr, bkg
        return snr

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
        this functio will default to the above equation under the assumption it
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

    def to_table(self):
        """Exports the Periodogram as an Astropy Table.

        Returns
        -------
        table : `astropy.table.Table` object
            An AstroPy Table with columns 'frequency', 'period', and 'power'.
        """
        return Table(data=(self.frequency, self.period, self.power),
                     names=('frequency', 'period', 'power'),
                     meta=self.meta)

    def copy(self):
        """Returns a copy of the Periodogram object.

        This method uses the `copy.deepcopy` function to ensure that all
        objects stored within the Periodogram are copied.

        Returns
        -------
        pg_copy : Periodogram
            A new `Periodogram` object which is a copy of the original.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        return('Periodogram(ID: {})'.format(self.targetid))

    def __getitem__(self, key):
        copy_self = self.copy()
        copy_self.frequency = self.frequency[key]
        copy_self.power = self.power[key]
        return copy_self

    def __add__(self, other):
        copy_self = self.copy()
        copy_self.power = copy_self.power + u.Quantity(other, self.power.unit)
        return copy_self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        copy_self = self.copy()
        copy_self.power = other - copy_self.power
        return copy_self

    def __mul__(self, other):
        copy_self = self.copy()
        copy_self.power = other * copy_self.power
        return copy_self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        copy_self = self.copy()
        copy_self.power = other / copy_self.power
        return copy_self

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def show_properties(self):
        """Prints a summary of the non-callable attributes of the Periodogram object.

        Prints in order of type (ints, strings, lists, arrays and others).
        Prints in alphabetical order.
        """
        attrs = {}
        for attr in dir(self):
            if not attr.startswith('_'):
                res = getattr(self, attr)
                if callable(res):
                    continue

                if isinstance(res, astropy.units.quantity.Quantity):
                    unit = res.unit
                    res = res.value
                    attrs[attr] = {'res': res}
                    attrs[attr]['unit'] = unit.to_string()
                else:
                    attrs[attr] = {'res': res}
                    attrs[attr]['unit'] = ''

                if attr == 'hdu':
                    attrs[attr] = {'res': res, 'type': 'list'}
                    for idx, r in enumerate(res):
                        if idx == 0:
                            attrs[attr]['print'] = '{}'.format(r.header['EXTNAME'])
                        else:
                            attrs[attr]['print'] = '{}, {}'.format(
                                attrs[attr]['print'], '{}'.format(r.header['EXTNAME']))
                    continue

                if isinstance(res, int):
                    attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'int'
                elif isinstance(res, float):
                    attrs[attr]['print'] = '{}'.format(np.round(res, 4))
                    attrs[attr]['type'] = 'float'
                elif isinstance(res, np.ndarray):
                    attrs[attr]['print'] = 'array {}'.format(res.shape)
                    attrs[attr]['type'] = 'array'
                elif isinstance(res, list):
                    attrs[attr]['print'] = 'list length {}'.format(len(res))
                    attrs[attr]['type'] = 'list'
                elif isinstance(res, str):
                    if res == '':
                        attrs[attr]['print'] = '{}'.format('None')
                    else:
                        attrs[attr]['print'] = '{}'.format(res)
                    attrs[attr]['type'] = 'str'
                elif attr == 'wcs':
                    attrs[attr]['print'] = 'astropy.wcs.wcs.WCS'.format(attr)
                    attrs[attr]['type'] = 'other'
                else:
                    attrs[attr]['print'] = '{}'.format(type(res))
                    attrs[attr]['type'] = 'other'

        output = Table(names=['Attribute', 'Description', 'Units'],
                       dtype=[object, object, object])
        idx = 0
        types = ['int', 'str', 'float', 'list', 'array', 'other']
        for typ in types:
            for attr, dic in attrs.items():
                if dic['type'] == typ:
                    output.add_row([attr, dic['print'], dic['unit']])
                    idx += 1
        print('lightkurve.Periodogram properties:')
        output.pprint(max_lines=-1, max_width=-1)


class SNRPeriodogram(Periodogram):
    """Defines a Signal-to-Noise Ratio (SNR) Periodogram class.

    This class is nearly identical to the standard :class:`Periodogram` class,
    but has different plotting defaults.
    """
    def __init__(self, *args, **kwargs):
        super(SNRPeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('SNRPeriodogram(ID: {})'.format(self.targetid))

    def plot(self, **kwargs):
        """Plot the SNR spectrum using matplotlib's `plot` method.
        See `Periodogram.plot` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments ot be passed to `Periodogram.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(SNRPeriodogram, self).plot(**kwargs)
        if 'ylabel' not in kwargs:
            ax.set_ylabel("Signal to Noise Ratio (SNR)")
        return ax

    def estimate_numax(self, numaxs=None):
        """Estimates the peak of the envelope of seismic oscillation modes,
        numax using an autocorrelation function. There are many papers on the
        topic of autocorrelation functions for estimating seismic parameters,
        including but not limited to: Roxburgh & Vorontsov (2006),
        Roxburgh (2009), Mosser & Appourchaux (2009), Huber et al. (2009),
        Verner & Roxburgh (2011) & Viani et al. (2019).

        We base this approach first and foremost off the 2D ACF numax estimation
        presented in Viani et al. (2019) and other papers above, but instead of
        using a moving window of fixed width, we use a moving window equal to
        one estimated full width half maximum (FWHM) either side of a central
        frequency, where the central frequency functions as numax in estimating
        the FWHM. This window is then mulitplied wiht a hanning window, which
        reduces power of peaks in the spectrum that do not follow the expected
        shape of a seismic mode envelope.

        The correlation (numpy.correlate) is typically given as:

        C[x, y] = sum( x * conj(y) ) .

        The autocorrelation power of a full spectrum with itself is then

        C = sum(s * s),

        where s is a window of the signal-to-noise spectrum.
        As FWHM is a function of numax, the autocorrelated spectrum will be
        larger for evaluations at larger numax. To avoid unfair weighting
        towards higher numax values, we rescale to a metric M as

        M = sqrt(C / len(C))

        We first create an array of sensible numax values based on the cadence
        of the timeseries. We then estimate the FWHM of the mode envelope at
        each numax, and calculate the ACF for this region. Near the true numax,
        the consistent power excess of the modes will increase the value of
        the ACF, highlighting the location of numax.

        We then covolve M with an Astropy Gaussian 1D Kernel with a standard
        deviation of 5 to smooth it. The frequency that results in the highest
        value of M is the detected numax.

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
            none is given, a sensible range will be chosen.

        Returns:
        --------
        numax : float
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        """

        numax,_,_,_,_,_ = self._estimate_numax(numaxs)
        return numax

    def plot_numax_diagnostics(self, numaxs=None, return_metric=False):
        """ Returns three diagnostic plots and an estimated value for numax.

        [1] The SNRPeriodogram plotted with a red line indicating the estimated
        numax value.

        [2] An image showing the 2D autocorrelation. On the y-axis is the
        frequency lag of the autocorrelation window. On the x-axis is the
        central frequency at which the autocorrelation was calculated. In the
        z-axis is the unitless autocorrelatin power. Shown in red is the
        estmated numax.

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

        return_metric : bool
            If True, returns the metric data shown in the lower diagnostic plot.

        Returns:
        --------
        numax : float
            The numax of the periodogram. In the units of the periodogram object
            frequency.
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        metric : ndarray
            The (unsmoothed) autocorrelation metric shown in the diagnostic plot.
            Only returned if `return_metric = True`.
        """
        numax, numaxrange, acf2d, window, metric, metric_smooth = self._estimate_numax(numaxs)

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots(3,sharex=True,figsize=(8.485, 12))
            self.plot(ax=ax[0])
            ax[0].set_ylabel(r'SNR')
            ax[0].set_title(r'SNR vs Frequency')
            ax[0].set_xlabel(None)

            windowarray = np.linspace(0, window, num=acf2d.shape[1])
            extent = (numaxrange[0],numaxrange[-1],windowarray[0],windowarray[-1])
            figsize = [8.485, 4]
            a = figsize[1]/figsize[0]
            b = (extent[3]-extent[2])/extent[1]

            ax[1].imshow(acf2d,cmap='Blues', aspect=a/b, origin='lower',extent=extent)
            ax[1].set_ylabel(r'Frequency lag [{}]'.format(self.frequency.unit.to_string('latex')))

            ax[2].plot(numaxrange,metric)
            ax[2].plot(numaxrange,metric_smooth)
            ax[2].set_xlabel("Frequency [{}]".format(self.frequency.unit.to_string('latex')))
            ax[2].set_ylabel(r'Correlation Metric')
            ax[0].axvline(numax.value,c='r', linewidth=2,alpha=.4)
            ax[1].axvline(numax.value,c='r', linewidth=2,alpha=.4)
            ax[2].axvline(numax.value,c='r', linewidth=2,alpha=.4,
                label=r'{:.1f} {}'.format(numax.value,
                                    self.frequency.unit.to_string('latex')))
            ax[2].legend()

        if return_metric:
            return numax, ax, metric
        else:
            return numax, ax

    def _estimate_numax(self, numaxs):
        """
        Helper function to perform the numax estimation for both the
        `estimate_numax()` and `plot_numax_diagnostics()` functions.

        For details, see the `estimate_numax()` function.
        """
        # Run some checks on the passed in numaxs
        if numaxs is not None:
            numaxs = u.Quantity(numaxs, self.frequency.unit).value
            fs = np.median(np.diff(self.frequency.value))
            if any(numaxs < fs):
                raise ValueError("A custom range of numaxs can not extend below"
                                "a single frequency bin.")
            if any(numaxs > np.nanmax(self.frequency.value)):
                raise ValueError("A custom range of numaxs can not extend above"
                                "the highest frequency value in the periodogram.")

        # Calcualte the window size
        if u.Quantity(self.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
            window = 250.
        else:
            window = 25.

        if numaxs is None:
            numaxs = np.arange(window/2, np.floor(np.nanmax(self.frequency.value)) - window/2, 1.)

        #We want to find the numax which returns in the highest autocorrelation
        #power, rescaled based on filter width
        fs = np.median(np.diff(self.frequency.value))

        metric = np.zeros(len(numaxs))
        acf2d = np.zeros([int(window/2/fs)*2,len(numaxs)])
        for idx, numax in enumerate(numaxs):
            acf = self._autocorrelate(numax, window=window)      #Return the acf at this numax
            acf2d[:,idx] = acf                                     #Store the 2D acf
            metric[idx] = (np.sum(np.abs(acf)) - 1 ) / len(acf)  #Store the max acf power normalised by the length

        # Smooth the data to find the peak
        g = Gaussian1DKernel(stddev=int(window/5))
        metric_smooth = convolve(metric, g)
        best_numax = numaxs[np.argmax(metric_smooth)]     #The highest value of the metric corresponds to numax

        return u.Quantity(best_numax, self.frequency.unit), \
                numaxs, acf2d, window, metric, metric_smooth

    def estimate_dnu(self, numax=None):
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
        mode envelope either side of the central frequency.

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
        The peak closest to an empirical estimate of dnu is taken as the true
        value. The peak finding algorithm is limited by a minimum spacing
        between peaks of 0.5 times the empirical value for dnu.

        Our empirical estimate for numax is taken from Stello et al. 2009 as

        dnu = 0.294 * numax^0.772

        If `numax` is None, a numax is calculated using the estimate_numax()
        function with default settings.

        NOTE: When plotting the acf, we exclude the first frequency lag bin, to
        make the relevant features on the plot clearer.

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
        dnu : float
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.
        """

        dnu,_,_,_,_ = self._estimate_dnu(numax)
        return dnu

    def plot_dnu_diagnostics(self, numax=None, return_metric=False):
        """ Returns one diagnostic plots and an estimated value for dnu.

        [1] Scaled correlation metrix vs frequecy lag of the autocorrelation
        window, with inset close up on the determined dnu and a line indicating
        the determined dnu.

        For details on the dnu estimation, see the `estimate_dnu()` function.
        The calculation performed is identical.

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
        dnu : float
            The average large frequency spacing of the seismic oscillation modes.
            In units of the periodogram frequency attribute.

        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.

        metric : ndarray
            The (unsmoothed) autocorrelation metric shown in the diagnostic plot.
            Only returned if `return_metric = True`.
        """
        dnu, lags, metric, peaks, sel = self._estimate_dnu(numax)

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots()
            # ax.plot(lags, acf/acf[0])
            ax.plot(lags[1:], metric[1:])
            ax.set_xlabel("Frequency Lag [{}]".format(self.frequency.unit.to_string('latex')))
            ax.set_ylabel(r'Scaled Correlation')
            ax.axvline(dnu.value,c='r', linewidth=2,alpha=.4)
            ax.set_title(r'Scaled Correlation vs Lag for a given $\nu_{\rm max}$')

            axin = inset_axes(ax, width="50%",height="50%", loc="upper right")
            axin.set_yticks([])
            axin.plot(lags[sel],metric[sel])
            axin.scatter(lags[sel][peaks], metric[sel][peaks],c='r',s=5)
            axin.axvline(dnu.value,c='r', linewidth=2,alpha=.4,
                label=r'{:.1f} {}'.format(dnu.value,
                                    self.frequency.unit.to_string('latex')))
            axin.legend(loc='best')

        if return_metric:
            return dnu, ax, metric
        else:
            return dnu, ax

    def _estimate_dnu(self, numax):
        """
        Helper function to perform the dnu estimation for both the
        `estimate_dnu()` and `plot_dnu_diagnostics()` functions.

        For details, see the `estimate_dnu()` function.
        """

        # Run some checks on the passed in numaxs
        if numax is not None:
            # Ensure input numax is in the correct units
            numax = u.Quantity(numax, self.frequency.unit)
            fs = np.median(np.diff(self.frequency.value))
            if numax.value < fs:
                raise ValueError("The input numax can not be lower than"
                                " a single frequency bin.")
            if numax.value > np.nanmax(self.frequency.value):
                raise ValueError("The input numax can not be higher than"
                                "the highest frequency value in the periodogram.")

        elif numax is None:
            #Estimate numax using the default settings
            numax = self.estimate_numax()

        #Calcluate dnu using the method by Stello et al. 2009
        #Make sure that this relation only ever happens in microhertz space
        dnu_emp = u.Quantity((0.294 * u.Quantity(numax, u.microhertz).value ** 0.772)*u.microhertz,
                            self.frequency.unit).value

        window = 2*int(np.floor(self._get_fwhm(numax.value)))
        aacf = self._autocorrelate(numax = numax.value, window=window)
        acf = (np.abs(aacf**2)/np.abs(aacf[0]**2)) / (3/(2*len(aacf)))
        fs = np.median(np.diff(self.frequency.value))
        lags = np.linspace(0., len(acf)*fs, len(acf))

        #Select a 25% region region around the empirical dnu
        sel = (lags > dnu_emp - .25*dnu_emp) & (lags < dnu_emp + .25*dnu_emp)

        #Run a peak finder on this region
        peaks, _ = find_peaks(acf[sel], distance=np.floor(dnu_emp/2. / fs))

        #Select the peak closest to the empirical value
        best_dnu = lags[sel][peaks][np.argmin(np.abs(lags[sel][peaks] - dnu_emp))]

        return u.Quantity(best_dnu, self.frequency.unit),\
                lags, acf, peaks, sel

    def _autocorrelate(self, numax, window=25.):
        """An autocorrelation function (ACF) for seismic mode envelopes.
        We autocorrelate the region one full-width-half-maximum (FWHM) of the
        mode envelope either side of the proposed numax.
        Before autocorrelating, it multiplies the section with a hanning
        window, which will increase the autocorrelation power if the region
        has a Gaussian shape, as we'd expect for seismic oscillations.

        Parameters:
        ----------
            numax : float
                The estimated position of the numax of the power spectrum. This
                is used to calculated the region autocorrelated with itself.

        Returns:
        --------
            acf : array-like
                The autocorrelation power calculated for the given numax
        """
        fs = np.median(np.diff(self.frequency.value))

        spread = int(window/2/fs)                           # Find the spread in indices
        x = int(numax / fs)                                 # Find the index value of numax
        p_sel = self.power[x-spread:x+spread].value         # Make the window selection

        C = np.correlate(p_sel, p_sel, mode='full')         #Correlated the resulting SNR space with itself
        C = C[len(p_sel)-1:]                                #Truncate the ACF
        return C

class LombScarglePeriodogram(Periodogram):
    """Subclass of :class:`Periodogram <lightkurve.periodogram.Periodogram>`
    representing a power spectrum generated using the Lomb Scargle method.
    """
    def __init__(self, *args, **kwargs):
        super(LombScarglePeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('LombScarglePeriodogram(ID: {})'.format(self.targetid))

    @staticmethod
    def from_lightcurve(lc, minimum_frequency=None, maximum_frequency=None,
                        minimum_period=None, maximum_period=None,
                        frequency=None, period=None,
                        nterms=1, nyquist_factor=1, oversample_factor=None,
                        freq_unit=None, normalization="amplitude",
                        **kwargs):
        """Creates a Periodogram from a LightCurve using the Lomb-Scargle method.

        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequency`
        parameter or a custom regular grid of periods using the `period`
        parameter.

        The sampling of the spectrum can be changed using the
        `oversample_factor` parameter. An oversampled spectrum
        (oversample_factor > 1) is useful for displaying the full details
        of the spectrum, allowing the frequencies and amplitudes to be
        measured directly from the plot itself, with no fitting required.
        This is recommended for most applications, with a value of 5 or
        10. On the other hand, an oversample_factor of 1 means the spectrum
        is critically sampled, where every point in the spectrum is
        independent of the others. This may be used when Lorentzians are to
        be fitted to modes in the power spectrum, in cases where the mode
        lifetimes are shorter than the time-base of the data (which is
        sometimes the case for solar-like oscillations). An
        oversample_factor of 1 is suitable for these stars because the
        modes are usually fully resolved. That is, the power from each mode
        is spread over a range of frequencies due to damping.  Hence, any
        small error from measuring mode frequencies by taking the maximum
        of the peak is negligible compared with the intrinsic linewidth of
        the modes.

        The `normalization` parameter will normalize the spectrum to either
        power spectral density ("psd") or amplitude ("amplitude"). Users
        doing asteroseismology on classical pulsators (e.g. delta Scutis)
        typically prefer `normalization="amplitude"` because "amplitude"
        has higher dynamic range (high and low peaks visible
        simultaneously), and we often want to read off amplitudes from the
        plot. If `normalization="amplitude"`, the default value for
        `oversample_factor` is set to 5 and `freq_unit` is 1/day.
        Alternatively, users doing asteroseismology on solar-like
        oscillators tend to prefer `normalization="psd"` because power
        density has a scaled axis that depends on the length of the
        observing time, and is used when we are interested in noise levels
        (e.g. granulation) and are looking at damped oscillations. If
        `normalization="psd"`, the default value for `oversample_factor` is
        set to 1 and `freq_unit` is set to microHz.  Default values of
        `freq_unit` and `oversample_factor` can be overridden. See Appendix
        A of Kjeldsen & Bedding, 1995 for a full discussion of
        normalization and measurement of oscillation amplitudes
        (http://adsabs.harvard.edu/abs/1995A%26A...293...87K).

        The parameter nterms controls how many Fourier terms are used in the
        model. Setting the Nyquist_factor to be greater than 1 will sample the
        space beyond the Nyquist frequency, which may introduce aliasing.

        The `freq_unit` parameter allows a request for alternative units in frequency
        space. By default frequency is in (1/day) and power in (amplitude
        (ppm)). Asteroseismologists for example may want frequency in (microHz)
        in which case they would pass `freq_unit=u.microhertz`.

        By default this method uses the LombScargle 'fast' method, which assumes
        a regular grid. If a regular grid of periods (i.e. an irregular grid of
        frequencies) it will use the 'slow' method. If nterms > 1 is passed, it
        will use the 'fastchi2' method for regular grids, and 'chi2' for
        irregular grids.

        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.

        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        minimum_frequency : float
            If specified, use this minimum frequency rather than one over the
            time baseline.
        maximum_frequency : float
            If specified, use this maximum frequency rather than nyquist_factor
            times the nyquist frequency.
        minimum_period : float
            If specified, use 1./minium_period as the maximum frequency rather
            than nyquist_factor times the nyquist frequency.
        maximum_period : float
            If specified, use 1./maximum_period as the minimum frequency rather
            than one over the time baseline.
        frequency :  array-like
            The regular grid of frequencies to use. If given a unit, it is
            converted to units of freq_unit. If not, it is assumed to be in
            units of freq_unit. This over rides any set frequency limits.
        period : array-like
            The regular grid of periods to use (as 1/period). If given a unit,
            it is converted to units of freq_unit. If not, it is assumed to be
            in units of 1/freq_unit. This overrides any set period limits.
        nterms : int
            Default 1. Number of terms to use in the Fourier fit.
        nyquist_factor : int
            Default 1. The multiple of the average Nyquist frequency. Is
            overriden by maximum_frequency (or minimum period).
        oversample_factor : int
            Default: None. The frequency spacing, determined by the time
            baseline of the lightcurve, is divided by this factor, oversampling
            the frequency space. This parameter is identical to the
            samples_per_peak parameter in astropy.LombScargle(). If
            normalization='amplitude', oversample_factor will be set to 5. If
            normalization='psd', it will be 1. These defaults can be
            overridden.
         freq_unit : `astropy.units.core.CompositeUnit`
            Default: None. The desired frequency units for the Lomb Scargle
            periodogram. This implies that 1/freq_unit is the units for period.
            With default normalization ('amplitude'), the freq_unit is set to
            1/day, which can be overridden. 'psd' normalization will set
            freq_unit to microhertz.
        normalization : 'psd' or 'amplitude'
            Default: `'amplitude'`. The desired normalization of the spectrum.
            Can be either power spectral density (`'psd'`) or amplitude
            (`'amplitude'`).
        kwargs : dict
            Keyword arguments passed to `astropy.stats.LombScargle()`

        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        # Input validation for spectrum type
        if normalization not in ('psd', 'amplitude'):
            raise ValueError("The `normalization` parameter must be one of "
                             "either 'psd' or 'amplitude'.")

        # Setting default frequency units
        if freq_unit is None:
            freq_unit = 1/u.day if normalization == 'amplitude' else u.microhertz

        # Default oversample factor
        if oversample_factor is None:
            oversample_factor = 5. if normalization == 'amplitude' else 1.

        if "min_period" in kwargs:
            warnings.warn("`min_period` keyword is deprecated, "
                          "please use `minimum_period` instead.",
                          LightkurveWarning)
            minimum_period = kwargs.pop("min_period", None)
        if "max_period" in kwargs:
            warnings.warn("`max_period` keyword is deprecated, "
                          "please use `maximum_period` instead.",
                          LightkurveWarning)
            maximum_period = kwargs.pop("max_period", None)
        if "min_frequency" in kwargs:
            warnings.warn("`min_frequency` keyword is deprecated, "
                          "please use `minimum_frequency` instead.",
                          LightkurveWarning)
            minimum_frequency = kwargs.pop("min_frequency", None)
        if "max_frequency" in kwargs:
            warnings.warn("`max_frequency` keyword is deprecated, "
                          "please use `maximum_frequency` instead.",
                          LightkurveWarning)
            maximum_frequency = kwargs.pop("max_frequency", None)

        # Make sure the lightcurve object is normalized
        lc = lc.normalize()

        # Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, minimum_period, maximum_period]):
            default_view = 'period'
        else:
            default_view = 'frequency'

        # If period and frequency keywords have both been set, throw an error
        if (not all(b is None for b in [period, minimum_period, maximum_period])) & \
           (not all(b is None for b in [frequency, minimum_frequency, maximum_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. '
                             'Please only use one.')

        if (~np.isfinite(lc.flux)).any():
            raise ValueError('Lightcurve contains NaN values. Use lc.remove_nans()'
                             ' to remove NaN values from a LightCurve.')

        # Hard coding that time is in days.
        time = lc.time.copy() * u.day

        # Approximate Nyquist Frequency and frequency bin width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(time))))
        fs = (1./(time[-1] - time[0])) / oversample_factor

        # Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        # Warn if there is confusing input
        if (frequency is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            log.warning("You have passed both a grid of frequencies "
                        "and min_frequency/maximum_frequency arguments; "
                        "the latter will be ignored.")
        if (period is not None) & (any([a is not None for a in [minimum_period, maximum_period]])):
            log.warning("You have passed a grid of periods "
                        "and minimum_period/maximum_period arguments; "
                        "the latter will be ignored.")

        # Tidy up the period stuff...
        if maximum_period is not None:
            # minimum_frequency MUST be none by this point.
            minimum_frequency = 1. / maximum_period
        if minimum_period is not None:
            # maximum_frequency MUST be none by this point.
            maximum_frequency = 1. / minimum_period
        # If the user specified a period, copy it into the frequency.
        if (period is not None):
            frequency = 1. / period

        # Do unit conversions if user input min/max frequency or period
        if frequency is None:
            if minimum_frequency is not None:
                minimum_frequency = u.Quantity(minimum_frequency, freq_unit)
            if maximum_frequency is not None:
                maximum_frequency = u.Quantity(maximum_frequency, freq_unit)
            if (minimum_frequency is not None) & (maximum_frequency is not None):
                if (minimum_frequency > maximum_frequency):
                    if default_view == 'frequency':
                        raise ValueError('minimum_frequency cannot be larger than maximum_frequency')
                    if default_view == 'period':
                        raise ValueError('minimum_period cannot be larger than maximum_period')
            # If nothing has been passed in, set them to the defaults
            if minimum_frequency is None:
                minimum_frequency = fs
            if maximum_frequency is None:
                maximum_frequency = nyquist * nyquist_factor

            # Create frequency grid evenly spaced in frequency
            frequency = np.arange(minimum_frequency.value, maximum_frequency.value, fs.to(freq_unit).value)

        # Convert to desired units
        frequency = u.Quantity(frequency, freq_unit)

        if nterms > 1:
            raise NotImplementedError('Increasing the number of terms is not implemented yet.')
        else:
            method = 'fast'

        if period is not None:
            method = 'slow'
            log.warning("You have passed an evenly-spaced grid of periods. "
                        "These are not evenly spaced in frequency space.\n"
                        "Method has been set to 'slow' to allow for this.")
        flux_scaling = 1e6
        if float(astropy.__version__[0]) >= 3:
            LS = LombScargle(time, lc.flux * flux_scaling,
                             nterms=nterms, normalization='psd', **kwargs)
            power = LS.power(frequency, method=method)
        else:
            LS = LombScargle(time, lc.flux * flux_scaling,
                             nterms=nterms, **kwargs)
            power = LS.power(frequency, method=method, normalization='psd')

        # Power spectral density
        if normalization == 'psd':
            # Rescale from the unnormalized  power output by Astropy's
            # Lomb-Scargle function to units of ppm^2 / [frequency unit]
            # that may be of more interest for asteroseismology.
            power *=  2./(len(time)*oversample_factor*fs) * (cds.ppm**2)

        # Amplitude spectrum
        elif normalization == 'amplitude':
            factor = np.sqrt(4./len(lc.time))
            power = np.sqrt(power) * factor
            # Units of ppm
            power *= cds.ppm

        # Periodogram needs properties
        return LombScarglePeriodogram(frequency=frequency, power=power, nyquist=nyquist,
                                      targetid=lc.targetid, label=lc.label,
                                      default_view=default_view)


class BoxLeastSquaresPeriodogram(Periodogram):
    """Subclass of :class:`Periodogram <lightkurve.periodogram.Periodogram>`
    representing a power spectrum generated using the Box Least Squares (BLS) method.
    """
    def __init__(self, *args, **kwargs):
        self.duration = kwargs.pop("duration", None)
        self.depth = kwargs.pop("depth", None)
        self.snr = kwargs.pop("snr", None)
        self._BLS_result = kwargs.pop("bls_result", None)
        self._BLS_object = kwargs.pop("bls_obj", None)

        self.transit_time = kwargs.pop("transit_time", None)
        self.time = kwargs.pop("time", None)
        self.flux = kwargs.pop("flux", None)
        self.time_unit = kwargs.pop("time_unit", None)
        super(BoxLeastSquaresPeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('BoxLeastSquaresPeriodogram(ID: {})'.format(self.targetid))

    @staticmethod
    def from_lightcurve(lc, **kwargs):
        """Creates a Periodogram from a LightCurve using the Box Least Squares (BLS) method."""
        # BoxLeastSquares was added to `astropy.stats` in AstroPy v3.1 and then
        # moved to `astropy.timeseries` in v3.2, which makes the import below
        # somewhat complicated.
        try:
            from astropy.timeseries import BoxLeastSquares
        except ImportError:
            try:
                from astropy.stats import BoxLeastSquares
            except ImportError:
                raise ImportError("BLS requires AstroPy v3.1 or later")

        # Validate user input for `lc`
        # (BoxLeastSquares will not work if flux or flux_err contain NaNs)
        lc = lc.remove_nans()
        if np.isfinite(lc.flux_err).all():
            dy = lc.flux_err
        else:
            dy = None

        # Validate user input for `duration`
        duration = kwargs.pop("duration", 0.25)
        if duration is not None and ~np.all(np.isfinite(duration)):
            raise ValueError("`duration` parameter contains illegal nan or inf value(s)")

        # Validate user input for `period`
        period = kwargs.pop("period", None)
        minimum_period = kwargs.pop("minimum_period", None)
        maximum_period = kwargs.pop("maximum_period", None)
        if period is not None and ~np.all(np.isfinite(period)):
            raise ValueError("`period` parameter contains illegal nan or inf value(s)")
        if minimum_period is None:
            if period is None:
                minimum_period = np.max([np.median(np.diff(lc.time)) * 4,
                                         np.max(duration) + np.median(np.diff(lc.time))])
            else:
                minimum_period = np.min(period)
        if maximum_period is None:
            if period is None:
                maximum_period = (np.max(lc.time) - np.min(lc.time)) / 3.
            else:
                maximum_period = np.max(period)

        # Validate user input for `time_unit`
        time_unit = (kwargs.pop("time_unit", "day"))
        if time_unit not in dir(u):
            raise ValueError('{} is not a valid value for `time_unit`'.format(time_unit))

        # Validate user input for `frequency_factor`
        frequency_factor = kwargs.pop("frequency_factor", 10)
        df = frequency_factor * np.min(duration) / (np.max(lc.time) - np.min(lc.time))**2
        npoints = int(((1/minimum_period) - (1/maximum_period))/df)
        if npoints > 1e7:
            raise ValueError('`period` contains {} points.'
                             'Periodogram is too large to evaluate. '
                             'Consider setting `frequency_factor` to a higher value.'
                             ''.format(np.round(npoints, 4)))
        elif npoints > 1e5:
            log.warning('`period` contains {} points.'
                        'Periodogram is likely to be large, and slow to evaluate. '
                        'Consider setting `frequency_factor` to a higher value.'
                        ''.format(np.round(npoints, 4)))

        # Create BLS object and run the BLS search
        bls = BoxLeastSquares(lc.time, lc.flux, dy)
        if period is None:
            period = bls.autoperiod(duration,
                                    minimum_period=minimum_period,
                                    maximum_period=maximum_period,
                                    frequency_factor=frequency_factor)
        result = bls.power(period, duration, **kwargs)
        if not isinstance(result.period, u.quantity.Quantity):
            result.period = u.Quantity(result.period, time_unit)
        if not isinstance(result.power, u.quantity.Quantity):
            result.power = result.power * u.dimensionless_unscaled
        if not isinstance(result.duration, u.quantity.Quantity):
            result.duration = u.Quantity(result.duration, time_unit)

        return BoxLeastSquaresPeriodogram(frequency=1. / result.period,
                                          power=result.power,
                                          default_view='period',
                                          label=lc.label,
                                          targetid=lc.targetid,
                                          transit_time=result.transit_time,
                                          duration=result.duration,
                                          depth=result.depth,
                                          bls_result=result,
                                          snr=result.depth_snr,
                                          bls_obj=bls,
                                          time=lc.time,
                                          flux=lc.flux,
                                          time_unit=time_unit)

    def compute_stats(self, period=None, duration=None, transit_time=None):
        """Computes commonly used vetting statistics for a transit model.

        See astropy.stats.bls docs for further details.

        Parameters
        ----------
        period : float or Quantity
            Period of the transits. Default is `period_at_max_power`
        duration : float or Quantity
            Duration of the transits. Default is `duration_at_max_power`
        transit_time : float or Quantity
            Transit midpoint of the transits. Default is `transit_time_at_max_power`

        Returns
        -------
        stats : dict
            Dictionary of vetting statistics
        """
        if period is None:
            period = self.period_at_max_power
            log.warning('No period specified. Using period at max power')
        if duration is None:
            duration = self.duration_at_max_power
            log.warning('No duration specified. Using duration at max power')
        if transit_time is None:
            transit_time = self.transit_time_at_max_power
            log.warning('No transit time specified. Using transit time at max power')
        return self._BLS_object.compute_stats(u.Quantity(period, 'd').value,
                                              u.Quantity(duration, 'd').value,
                                              u.Quantity(transit_time, 'd').value)

    def get_transit_model(self, period=None, duration=None, transit_time=None):
        """Computes the transit model using the BLS, returns a lightkurve.LightCurve

        See astropy.stats.bls docs for further details.

        Parameters
        ----------
        period : float or Quantity
            Period of the transits. Default is `period_at_max_power`
        duration : float or Quantity
            Duration of the transits. Default is `duration_at_max_power`
        transit_time : float or Quantity
            Transit midpoint of the transits. Default is `transit_time_at_max_power`

        Returns
        -------
        model : lightkurve.LightCurve
            Model of transit
        """
        from .lightcurve import LightCurve

        if period is None:
            period = self.period_at_max_power
            log.warning('No period specified. Using period at max power')
        if duration is None:
            duration = self.duration_at_max_power
            log.warning('No duration specified. Using duration at max power')
        if transit_time is None:
            transit_time = self.transit_time_at_max_power
            log.warning('No transit time specified. Using transit time at max power')

        model_flux = self._BLS_object.model(self.time, u.Quantity(period, 'd').value,
                                            u.Quantity(duration, 'd').value,
                                            u.Quantity(transit_time, 'd').value)
        model = LightCurve(self.time, model_flux, label='Transit Model Flux')
        return model

    def get_transit_mask(self, period=None, duration=None, transit_time=None):
        """Computes the transit mask using the BLS, returns a lightkurve.LightCurve

        True where there are no transits.

        Parameters
        ----------
        period : float or Quantity
            Period of the transits. Default is `period_at_max_power`
        duration : float or Quantity
            Duration of the transits. Default is `duration_at_max_power`
        transit_time : float or Quantity
            Transit midpoint of the transits. Default is `transit_time_at_max_power`

        Returns
        -------
        mask : np.array of Bool
            Mask that removes transits. Mask is True where there are no transits.
        """
        model = self.get_transit_model(period=period, duration=duration, transit_time=transit_time)
        return model.flux == np.median(model.flux)

    @property
    def transit_time_at_max_power(self):
        """Returns the transit time corresponding to the highest peak in the periodogram."""
        return self.transit_time[np.nanargmax(self.power)]

    @property
    def duration_at_max_power(self):
        """Returns the duration corresponding to the highest peak in the periodogram."""
        return self.duration[np.nanargmax(self.power)]

    @property
    def depth_at_max_power(self):
        """Returns the depth corresponding to the highest peak in the periodogram."""
        return self.depth[np.nanargmax(self.power)]

    def plot(self, **kwargs):
        """Plot the BoxLeastSquaresPeriodogram spectrum using matplotlib's `plot` method.
        See `Periodogram.plot` for details on the accepted arguments.

        Parameters
        ----------
        kwargs : dict
            Dictionary of arguments ot be passed to `Periodogram.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        ax = super(BoxLeastSquaresPeriodogram, self).plot(**kwargs)
        if 'ylabel' not in kwargs:
            ax.set_ylabel("BLS Power")
        return ax

    def flatten(self, **kwargs):
        raise NotImplementedError('`flatten` is not implemented for `BoxLeastSquaresPeriodogram`.')

    def smooth(self, **kwargs):
        raise NotImplementedError('`smooth` is not implemented for `BoxLeastSquaresPeriodogram`. ')
