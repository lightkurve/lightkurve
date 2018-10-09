"""Defines Periodogram"""
from __future__ import division, print_function

import copy
import os
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt
import astropy
from astropy.table import Table
from astropy.io import fits
from astropy.stats import LombScargle
from scipy.ndimage.filters import gaussian_filter
from astropy import __version__
from scipy import interpolate


"""This module lets us attack a unit to a value or an array of values. This
allows us to keep track of what units our data are in, and easily switch
between different units. The cds module just contains some additional units not
in the standard units module, such as parts per million (ppm)."""
from astropy import units as u
from astropy.units import cds

from . import PACKAGEDIR, MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['Periodogram']


class Periodogram(object):
    """The Periodogram class represents a power spectrum, with values of
    frequency on the x-axis (in any frequency units) and values of power on the
    y-axis (in units of ppm^2 / [frequency units]). When calculated using a
    Lomb Scargle periodogram, it has additional attributes used in the calculation,
    such as `nyquist` and `frequency_spacing`.

    Attributes
    ----------
    frequency : array-like
        List of frequencies with associated astropy unit.
    power : array-like
        The power-spectral-density of the Fourier timeseries, in units of
        ppm^2 / freq_unit, where freq_unit is the unit of the frequency
        attribute.
    nyquist : float
        The Nyquist frequency of the lightcurve. In units of freq_unit, where
        freq_unit is the unit of the frequency attribute.
    frequency_spacing : float
        The frequency spacing of the periodogram. In units of freq_unit, where
        freq_unit is the unit of the frequency attribute.
    targetid : str
        Identifier of the target.
    label : str
        Human-friendly object label, e.g. "KIC 123456789"
    meta : dict
        Free-form metadata associated with the Periodogram.
    """
    def __init__(self, frequency, power,
                nyquist=None, frequency_spacing=None,
                label=None, targetid=None, meta={}):

        if not isinstance(power, u.quantity.Quantity):
            raise ValueError('Power must have units.')

        if not isinstance(frequency, u.quantity.Quantity):
            raise ValueError('Frequency must have units.')

        # Must have frequency units
        try:
             frequency.to(u.Hz)
        except u.UnitConversionError:
            raise ValueError('Frequency must be in units of 1/time.')

        if frequency.shape[0] <= 1:
            raise ValueError('Frequency and power must have a length greater than 1.')

        if frequency.shape != power.shape:
            raise ValueError('Frequency and power must be the same length.')

        self.frequency = frequency
        self.power = power
        self.nyquist = nyquist
        self.frequency_spacing = frequency_spacing
        self.label = label
        self.targetid = targetid
        self.meta = meta


    @property
    def period(self):
        """Returns list of periods (1 / frequency) with associated astropy unit
        """
        return (1./self.frequency)

    @property
    def max_power(self):
        """Returns the power of the highest peak in the periodogram."""
        return np.nanmax(self.power)

    @property
    def frequency_at_max_power(self):
        """Returns the frequency corresponding to the highest power in the
        periodogram"""
        return self.frequency[np.nanargmax(self.power)]
    @property
    def period_at_max_power(self):
        """Returns the period corresponding to the highest power in the
        periodogram."""
        return 1./self.frequency_at_max_power

    @staticmethod
    def from_lightcurve(lc, nterms=1, nyquist_factor=1, oversample_factor=1,
                        min_frequency=None, max_frequency=None,
                        min_period=None, max_period=None,
                        frequency=None, period=None,
                        freq_unit=1/u.day, **kwargs):
        """Creates a Periodogram object from a LightCurve instance using
        the Lomb-Scargle method.
        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequency`
        parameter or a custom regular grid of periods using the `period`
        parameter.

        The the spectrum can be oversampled by increasing the oversample_factor
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
        #Makes sure the lightcurve object is normalised
        lc = lc.normalize()

        #Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, min_period, max_period]):
            format = 'period'
        else:
            format = 'frequency'

        # If period and frequency keywords have both been set, throw an error
        if (not all(b is None for b in [period, min_period, max_period])) &\
            (not all(b is None for b in [frequency, min_frequency, max_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. '
                             'Please only use one.')

        if (~np.isfinite(lc.flux)).any():
            raise ValueError('Lightcurve contains NaN values. Use lc.remove_nans()'
                             ' to remove NaN values from a LightCurve.')

        # Hard coding that time is in days.
        time = lc.time.copy() * u.day

        #Calculate Nyquist Frequency and frequency bin eidth in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(time))))
        fs = (1./(time[-1] - time[0])) / oversample_factor

        #Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        # Warn if there is confusing input
        if (frequency is not None) & (any([a is not None for a in [min_frequency, max_frequency]])):
            log.warning('You have passed a grid of frequencies, which overrides any period/frequency limit kwargs.')
        if (period is not None) & (any([a is not None for a in [min_period, max_period]])):
            log.warning('You have passed a grid of periods, which overrides any period/frequency limit kwargs.')

        # Tidy up the period stuff...
        if max_period is not None:
            # min_frequency MUST be none by this point.
            min_frequency = 1./max_period
        if min_period is not None:
            # max_frequency MUST be none by this point.
            max_frequency = 1./min_period
#        # If the user specified a period, copy it into the frequency.
        if (period is not None):
            frequency = 1./period

        # Do unit conversions if user input min/max frequency or period
        if frequency is None:
            if min_frequency is not None:
                min_frequency = u.Quantity(min_frequency, freq_unit)
            if max_frequency is not None:
                max_frequency = u.Quantity(max_frequency, freq_unit)
            if (min_frequency is not None) & (max_frequency is not None):
                if (max_frequency <= min_frequency):
                    if format == 'frequency':
                        raise ValueError('User input max frequency is smaller than or equal to min frequency.')
                    if format == 'period':
                        raise ValueError('User input max period is smaller than or equal to min period.')
            #If nothing has been passed in, set them to the defaults
            if min_frequency is None:
                min_frequency = fs
            if max_frequency is None:
                max_frequency = nyquist * nyquist_factor

            #Create frequency grid evenly spaced in frequency
            frequency = np.arange(min_frequency.value, max_frequency.value, fs.to(freq_unit).value)

        #Convert to desired units
        frequency = u.Quantity(frequency, freq_unit)

        if nterms > 1:
            raise NotImplementedError('Increasing the number of terms is not yet implemented.')
        else:
            method='fast'

        if period is not None:
            method = 'slow'
            log.warning('You have passed an evenly-spaced grid of periods. These are not evenly spaced in frequency space.\n Method has been set to "slow" to allow for this.')

        if float(__version__[0]) >= 3:
            LS = LombScargle(time, lc.flux * 1e6,
                                nterms=nterms, normalization='psd', **kwargs)
            power = LS.power(frequency, method=method)
        else:
            LS = LombScargle(time, lc.flux * 1e6,
                                nterms=nterms, **kwargs)
            power = LS.power(frequency, method=method, normalization='psd')

        #Normalise the according to Parseval's theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        power = power * (cds.ppm**2)

        #Rescale power to units of ppm^2 / [frequency unit]
        power = power / fs


        ### Periodogram needs properties
        return Periodogram(frequency=frequency, power=power,
                            nyquist=nyquist, frequency_spacing=fs,
                            targetid=lc.targetid, label=lc.label)

    def bin(self, binsize=10, method='mean'):
        """Smooths the powerspectrum using a moving median filter.

        Parameters
        ----------
        binsize : int
            Default 10. The factor by which to bin the power spectrum, in the
            sense that the power spectrum will be smoothed by taking the mean
            in bins of size N / binsize, where N is the length of the
            original periodogram.
        method : str
            Method to use for binning. Default is mean.

        Returns
        -------
        smooth_periodogram : a `Periodogram` object
            Returns a `Periodogram` object which has been smoothed in bins of
            width `binsize`.
        """
        if binsize < 1:
            raise ValueError('The smooth factor must be greater than 1.')

        #Calculating the length of the smoothed array
        m = int(len(self.power) / binsize)
        if method == 'mean':
            smooth_freq = self.frequency[:m*binsize].reshape((m, binsize)).mean(1)
            smooth_power = self.power[:m*binsize].reshape((m, binsize)).mean(1)
        elif method == 'median':
            smooth_freq = np.nanmedian(self.frequency[:m*binsize].reshape((m, binsize)), axis=1)
            smooth_power = np.nanmedian(self.power[:m*binsize].reshape((m, binsize)), axis=1)
        else:
            raise ValueError('No such method as `{}`'.format(method))

        smooth_pg = copy.deepcopy(self)
        smooth_pg.frequency = smooth_freq
        smooth_pg.power = smooth_power
        return smooth_pg

    def plot(self, scale='linear', ax=None, xlabel=None, ylabel=None, title='',
                 style='lightkurve', format='frequency', unit=None, **kwargs):

        """Plots the periodogram.

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
            #Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend()
            ax.set_yscale(scale)
            ax.set_xscale(scale)
            ax.set_title(title)
        return ax

    def _estimate_background(self, log_width=0.01):
        """Estimates background noise of the power spectrum, via moving filter
        in log10 space. The filter defines a bin centered at a value x0 with a
        spread of log_width either side. The median of the power in this bin
        will be added to all indices within the bin in an empty array, `bkg`.
        The bin then moves along in a step of x0 + 0.5 * log_width. This means
        that each index will contain the sum of multiple medians of bins that
        index is included in. To normalize this, we divide the background value
        in each index by the number of median values that were added to that
        index.

        Parameters
        ----------
        log_width : float
            Default 0.01. The width of the filter in log10 space.

        Returns
        -------
        bkg : array-like
            An estimate of the noise background of the power spectrum. Has the
            same units as the `power` attribute.
        """
        if isinstance(self.frequency, astropy.units.quantity.Quantity):
            f = self.frequency.value
        else:
            f = self.frequency
        if isinstance(self.power, astropy.units.quantity.Quantity):
            p = self.power.value
        else:
            p = self.power

        count = np.zeros(len(f), dtype=int)
        bkg = np.zeros_like(f)
        x0 = np.log10(f[0])
        while x0 < np.log10(f[-1]):
            m = np.abs(np.log10(f) - x0) < log_width
            if len(bkg[m] > 0):
                bkg[m] += np.nanmedian(p[m])
                count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

    def flatten(self, log_width=0.01, return_trend=False):
        """Calculates the Signal-To-Noise spectrum of the power spectrum by
        dividing the power through by a background estimated using a moving
        filter in log10 space.

        Parameters
        ----------
        log_width : float
            Default 0.01. The width of the filter in log10 space. Kwarg for the
            Periodogram.estimate_background() function.

        Returns
        -------
        snr_spectrum: a `Periodogram` object
            Returns a periodogram object where the power is an estimate of the
            signal-to-noise of the spectrum, assuming a simple estimate of the
            noise background using a moving filter in log10 space.
        """
        bkg = u.Quantity(self._estimate_background(log_width=log_width), self.power.unit)
        snr_pg = self / bkg
        if return_trend:
            return Periodogram(snr_pg.frequency, bkg, nyquist = self.nyquist,
                                    frequency_spacing = self.frequency_spacing, targetid=self.targetid,
                                    label=self.label,
                                    meta = self.meta)
        return SNR_Periodogram(snr_pg.frequency, snr_pg.power, nyquist = self.nyquist,
                                frequency_spacing = self.frequency_spacing, targetid=self.targetid,
                                label=self.label,
                                meta = self.meta)

    def __repr__(self):
        return('Periodogram(ID: {})'.format(self.targetid))

    def __getitem__(self, key):
        copy_self = copy.copy(self)
        copy_self.frequency = self.frequency[key]
        copy_self.power = self.power[key]
        return copy_self

    def __add__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = copy_self.power + u.Quantity(other, self.power.unit)
        return copy_self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other - copy_self.power
        return copy_self

    def __mul__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other * copy_self.power
        return copy_self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1./other)

    def __rtruediv__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = other / copy_self.power
        return copy_self

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def properties(self):
        '''Print out a description of each of the non-callable attributes of a
        Periodogram object, as well as those of the LightCurve object it was
        made with.
        Prints in order of type (ints, strings, lists, arrays and others)
        Prints in alphabetical order.'''
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


class SNR_Periodogram(Periodogram):
    """Defines a periodogram with different plotting defaults"""
    def __init__(self, *args, **kwargs):
        super(SNR_Periodogram, self).__init__(*args, **kwargs)

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
        ax = super(SNR_Periodogram, self).plot(**kwargs)
        if 'ylabel' not in kwargs:
            ax.set_ylabel("Signal to Noise Ratio (SNR)")
        return ax
