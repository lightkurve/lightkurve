"""Defines the Periodogram class and associated tools."""
from __future__ import division, print_function

import copy
import logging

import numpy as np
from matplotlib import pyplot as plt

import astropy
from astropy.table import Table
from astropy.stats import LombScargle
from astropy import __version__
from astropy import units as u
from astropy.units import cds
from astropy.convolution import convolve, Box1DKernel

from . import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['Periodogram']


class Periodogram(object):
    """Class to represents a power spectrum, i.e. frequency vs power.

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
    targetid : str, optional
        Identifier of the target.
    label : str, optional
        Human-friendly object label, e.g. "KIC 123456789".
    meta : dict, optional
        Free-form metadata associated with the Periodogram.
    """
    def __init__(self, frequency, power, nyquist=None, label=None,
                 targetid=None, meta={}):
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
            format = 'period'
        else:
            format = 'frequency'

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
                    if format == 'frequency':
                        raise ValueError('min_frequency cannot be larger than max_frequency')
                    if format == 'period':
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

        if float(__version__[0]) >= 3:
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
        return Periodogram(frequency=frequency, power=power, nyquist=nyquist,
                           targetid=lc.targetid, label=lc.label)

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

            box_kernel = Box1DKernel(np.ceil(filter_width/fs))
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
            while x0 < np.log10(self.frequency[-1].value):
                m = np.abs(np.log10(self.frequency.value) - x0) < filter_width
                if len(bkg[m] > 0):
                    bkg[m] += np.nanmedian(self.power[m].value)
                    count[m] += 1
                x0 += 0.5 * filter_width
            bkg /= count
            smooth_pg = self.copy()
            smooth_pg.power = u.Quantity(bkg, self.power.unit)
            return smooth_pg

    def plot(self, scale='linear', ax=None, xlabel=None, ylabel=None, title='',
             style='lightkurve', format='frequency', unit=None, **kwargs):
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
        format : str
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

        if unit is None:
            unit = self.frequency.unit
            if format == 'period':
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
            if format.lower() == 'frequency':
                ax.plot(self.frequency.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Frequency [{}]".format(unit.to_string('latex'))
            elif format.lower() == 'period':
                ax.plot(self.period.to(unit), self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Period [{}]".format(unit.to_string('latex'))
            else:
                raise ValueError('{} is not a valid plotting format'.format(format))
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

        Returns
        -------
        snr_spectrum : `Periodogram` object
            Returns a periodogram object where the power is an estimate of the
            signal-to-noise of the spectrum, creating by dividing the powers
            with a simple estimate of the noise background using a smoothing filter.
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

    def properties(self):
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

        output = Table(names=['Attribute', 'Description', 'Units'], dtype=[object, object, object])
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
