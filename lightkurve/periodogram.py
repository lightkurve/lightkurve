"""Defines the Periodogram class and associated tools."""
from __future__ import division, print_function

import copy
import logging
import math

import numpy as np
from matplotlib import pyplot as plt

import astropy
from astropy.table import Table
from astropy.stats import LombScargle
from astropy import units as u
from astropy.units import cds
from astropy.convolution import convolve, Box1DKernel

from . import MPLSTYLE

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
        # Default view must be "frequency" or "period"
        allowed_views = ["frequency", "period"]
        if default_view not in allowed_views:
            raise ValueError(("Unrecognized default_view '{0}'\n"
                              "allowed values are: {1}")
                             .format(default_view, allowed_views))

        self.frequency = frequency
        self.power = power
        self.nyquist = nyquist
        self.label = label
        self.targetid = targetid
        self.default_view = default_view
        self.meta = meta

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

        if view is None:
            view = self.default_view

        if unit is None:
            unit = self.frequency.unit
            if view == 'period':
                unit = self.period.unit

        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        if ylabel is None:
            ylabel = "Power Spectral Density [{}]".format(self.power.unit.to_string('latex'))

        # This will need to be fixed with housekeeping. Self.label currently doesnt exist.
        if ('label' not in kwargs) and ('label' in dir(self)):
            kwargs['label'] = self.label

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot frequency and power
            if view.lower() == 'frequency':
                ax.plot(self.frequency.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Frequency [{}]".format(unit.to_string('latex'))
            elif view.lower() == 'period':
                ax.plot(self.period.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Period [{}]".format(unit.to_string('latex'))
            else:
                raise ValueError('{} is not a valid plotting view'.format(view))
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


class LombScarglePeriodogram(Periodogram):
    """Subclass of :class:`Periodogram <lightkurve.periodogram.Periodogram>`
    representing a power spectrum generated using the Lomb Scargle method.
    """
    def __init__(self, *args, **kwargs):
        super(LombScarglePeriodogram, self).__init__(*args, **kwargs)

    def __repr__(self):
        return('LombScarglePeriodogram(ID: {})'.format(self.targetid))

    @staticmethod
    def from_lightcurve(lc, min_frequency=None, max_frequency=None,
                        min_period=None, max_period=None,
                        frequency=None, period=None,
                        nterms=1, nyquist_factor=1, oversample_factor=1,
                        freq_unit=1/u.day, **kwargs):
        """Creates a Periodogram from a LightCurve using the Lomb-Scargle method.

        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequency`
        parameter or a custom regular grid of periods using the `period`
        parameter.

        The spectrum can be oversampled by increasing the oversample_factor
        parameter. The parameter nterms controls how many Fourier terms are used
        in the model. Note that many terms could lead to spurious peaks. Setting
        the Nyquist_factor to be greater than 1 will sample the space beyond the
        Nyquist frequency, which may introduce aliasing.

        The unit parameter allows a request for alternative units in frequency
        space. By default frequency is in (1/day) and power in (ppm^2 * day).
        Asteroseismologists for example may want frequency in (microHz) and
        power in (ppm^2 / microHz), in which case they would pass
        `unit = u.microhertz` where `u` is `astropy.units`

        By default this method uses the LombScargle 'fast' method, which assumes
        a regular grid. If a regular grid of periods (i.e. an irregular grid of
        frequencies) it will use the 'slow' method. If nterms > 1 is passed, it
        will use the 'fastchi2' method for regular grids, and 'chi2' for
        irregular grids. The normalizatin of the Lomb Scargle periodogram is
        fixed to `psd`, and cannot be overridden.

        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.

        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        min_frequency : float
            If specified, use this minimum frequency rather than one over the
            time baseline.
        max_frequency : float
            If specified, use this maximum frequency rather than nyquist_factor
            times the nyquist frequency.
        min_period : float
            If specified, use 1./minium_period as the maximum frequency rather
            than nyquist_factor times the nyquist frequency.
        max_period : float
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
            The frequency spacing, determined by the time baseline of the
            lightcurve, is divided by this factor, oversampling the frequency
            space. This parameter is identical to the samples_per_peak parameter
            in astropy.LombScargle()
        freq_unit : `astropy.units.core.CompositeUnit`
            Default: 1/u.day. The desired frequency units for the Lomb Scargle
            periodogram. This implies that 1/freq_unit is the units for period.
        kwargs : dict
            Keyword arguments passed to `astropy.stats.LombScargle()`

        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        # Make sure the lightcurve object is normalized
        lc = lc.normalize()

        # Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, min_period, max_period]):
            view = 'period'
        else:
            view = 'frequency'

        # If period and frequency keywords have both been set, throw an error
        if (not all(b is None for b in [period, min_period, max_period])) & \
           (not all(b is None for b in [frequency, min_frequency, max_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. '
                             'Please only use one.')

        if (~np.isfinite(lc.flux)).any():
            raise ValueError('Lightcurve contains NaN values. Use lc.remove_nans()'
                             ' to remove NaN values from a LightCurve.')

        # Hard coding that time is in days.
        time = lc.time.copy() * u.day

        # Calculate Nyquist Frequency and frequency bin width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(time))))
        fs = (1./(time[-1] - time[0])) / oversample_factor

        # Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        # Warn if there is confusing input
        if (frequency is not None) & (any([a is not None for a in [min_frequency, max_frequency]])):
            log.warning("You have passed both a grid of frequencies "
                        "and min_frequency/max_frequency arguments; "
                        "the latter will be ignored.")
        if (period is not None) & (any([a is not None for a in [min_period, max_period]])):
            log.warning("You have passed a grid of periods "
                        "and min_period/max_period arguments; "
                        "the latter will be ignored.")

        # Tidy up the period stuff...
        if max_period is not None:
            # min_frequency MUST be none by this point.
            min_frequency = 1. / max_period
        if min_period is not None:
            # max_frequency MUST be none by this point.
            max_frequency = 1. / min_period
        # If the user specified a period, copy it into the frequency.
        if (period is not None):
            frequency = 1. / period

        # Do unit conversions if user input min/max frequency or period
        if frequency is None:
            if min_frequency is not None:
                min_frequency = u.Quantity(min_frequency, freq_unit)
            if max_frequency is not None:
                max_frequency = u.Quantity(max_frequency, freq_unit)
            if (min_frequency is not None) & (max_frequency is not None):
                if (min_frequency > max_frequency):
                    if view == 'frequency':
                        raise ValueError('min_frequency cannot be larger than max_frequency')
                    if view == 'period':
                        raise ValueError('min_period cannot be larger than max_period')
            # If nothing has been passed in, set them to the defaults
            if min_frequency is None:
                min_frequency = fs
            if max_frequency is None:
                max_frequency = nyquist * nyquist_factor

            # Create frequency grid evenly spaced in frequency
            frequency = np.arange(min_frequency.value, max_frequency.value, fs.to(freq_unit).value)

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

        if float(astropy.__version__[0]) >= 3:
            LS = LombScargle(time, lc.flux * 1e6,
                             nterms=nterms, normalization='psd', **kwargs)
            power = LS.power(frequency, method=method)
        else:
            LS = LombScargle(time, lc.flux * 1e6,
                             nterms=nterms, **kwargs)
            power = LS.power(frequency, method=method, normalization='psd')

        # Normalise the according to Parseval's theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        power = power * (cds.ppm**2)

        # Rescale power to units of ppm^2 / [frequency unit]
        power = power / fs

        # Periodogram needs properties
        return LombScarglePeriodogram(frequency=frequency, power=power, nyquist=nyquist,
                                      targetid=lc.targetid, label=lc.label)


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
        time_unit = (kwargs.pop("time_unit", "day"))
        if time_unit not in dir(u):
            raise ValueError('{} is not a valid unit for time.'.format(time_unit))

        try:
            from astropy.stats import BoxLeastSquares
        except ImportError:
            raise Exception("BLS requires AstroPy v3.1 or later")

        # BoxLeastSquares will not work if flux or flux_err contain NaNs
        lc = lc.remove_nans()
        if np.isfinite(lc.flux_err).all():
            dy = lc.flux_err
        else:
            dy = None

        bls = BoxLeastSquares(lc.time, lc.flux, dy)
        duration = kwargs.pop("duration", 0.25)
        if hasattr(duration, '__iter__'):
            raise ValueError('`duration` must be a single value.')
        minimum_period = kwargs.pop("minimum_period", None)
        maximum_period = kwargs.pop("maximum_period", None)
        period = kwargs.pop("period", None)
        if minimum_period is None:
            if 'period' in kwargs:
                minimum_period = period.min()
            else:
                minimum_period = np.max([np.median(np.diff(lc.time)) * 4,
                                         duration + np.median(np.diff(lc.time))])
        if maximum_period is None:
            if 'period' in kwargs:
                maximum_period = period.max()
            else:
                maximum_period = (np.max(lc.time) - np.min(lc.time)) / 3.

        frequency_factor = kwargs.pop("frequency_factor", 10)
        df = frequency_factor * duration / (np.max(lc.time) - np.min(lc.time))**2
        npoints = int(((1/minimum_period) - (1/maximum_period))/df)

        # Too many points
        if npoints > 1e5:
            log.warning('`period` contains {} points.'
                        'Periodogram is likely to be large, and slow to evaluate. '
                        'Consider setting `frequency_factor` to a higher value.'
                        ''.format(np.round(npoints, 4)))

        # Way too many points
        if npoints > 1e7:
            raise ValueError('`period` contains {} points.'
                             'Periodogram is too large to evaluate. '
                             'Consider setting `frequency_factor` to a higher value.'
                             ''.format(np.round(npoints, 4)))

        period = kwargs.pop("period",
                            bls.autoperiod(duration,
                                           minimum_period=minimum_period,
                                           maximum_period=maximum_period,
                                           frequency_factor=frequency_factor))

        result = bls.power(period, duration, **kwargs)
        if not isinstance(result.period, u.quantity.Quantity):
            result.period = u.Quantity(result.period, time_unit)
        if not isinstance(result.power, u.quantity.Quantity):
            result.power = result.power * u.dimensionless_unscaled

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
