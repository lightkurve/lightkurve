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
from astropy.units import cds
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate


"""This module lets us attack a unit to a value or an array of values. This
allows us to keep track of what units our data are in, and easily switch
between different units."""
from astropy import units as u

from . import PACKAGEDIR, MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['Periodogram', 'estimate_mass', 'estimate_radius',
            'estimate_mean_density', 'stellar_params', 'standardize_units']

###Add uncertainties <- review these [astero]
numax_s = 3090.0 # Huber et al 2013
err_numax_s = 30.
deltanu_s = 135.1
err_deltanu_s = 0.1
teff_s = 5777.0


class Periodogram(object):
    ###Update this [houseekeping]
    """The Periodogram class represents a power spectrum, with values of
    frequency on the x-axis (in any frequency units) and values of power on the
    y-axis (in units of ppm^2 / [frequency units]). When calculated using a
    Lomb Scargle periodogram, it has additional attributes used in the calculation,
    such as `nyquist` and `frqeuency_spacing`.

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
    def from_lightcurve(lc, nterms = 1, nyquist_factor = 1, oversample_factor = 1,
                        min_frequency = None, max_frequency = None,
                        min_period = None, max_period = None,
                        frequency = None, period = None,
                        freq_unit = 1/u.day, **kwargs):
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
        #Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [period, min_period, max_period]):
            format = 'period'
        else: format = 'frequency'

        #Check input consistency
        if (not all(b is None for b in [period, min_period, max_period])) &\
            (not all(b is None for b in [frequency, min_frequency, max_frequency])):
            raise ValueError('You have input keyword arguments for both frequency and period. Please only use one or the other.')

        #Calculate Nyquist frequency & Frequency Bin Width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(lc.time))*u.day))
        fs = (1./((np.nanmax(lc.time - lc.time[0]))*u.day)) / oversample_factor

        #Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        if frequency is not None:
            log.warning('You have passed a grid of frequencies, which overrides any period/frequency limit kwargs.')

        #Check if period has been passed
        if period is not None:
            log.warning('You have passed a grid of periods, which overrides any period/frequency limit kwargs.')
            frequency = np.sort(1./period)
        if max_period is not None:
            min_frequency = 1./max_period
        if min_period is not None:
            max_frequency = 1./min_period

        #Do unit conversions if user input min/max frequency or period
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
            frequency = np.arange(min_frequency.value, max_frequency.value, fs.value)

        #Set in/convert to desired units
        frequency = u.Quantity(frequency, freq_unit)

        if nterms > 1:
            log.warning('Nterms has been set larger than 1. Method has been set to `fastchi2`')
            method = 'fastchi2'
            if period is not None:
                method = 'chi2'
                log.warning('You have passed an eventy-spaced grid of periods. These are not evenly spaced in frequency space.\n Method has been set to "chi2" to allow for this.')
        else:
            method='fast'
            if period is not None:
                method = 'slow'
                log.warning('You have passed an evenly-spaced grid of periods. These are not evenly spaced in frequency space.\n Method has been set to "slow" to allow for this.')


        LS = LombScargle(lc.time * u.day, lc.flux * 1e6,
                            nterms=nterms, normalization='psd', **kwargs)
        power = LS.power(frequency, method=method)

        #Normalise the according to Parseval's theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        #Rescale power to units of ppm^2 / [frequency unit]
        power = power / fs.value

        ### Periodogram needs properties
        return Periodogram(frequency=frequency, power=power,
                            nyquist=nyquist, frequency_spacing=fs,
                            targetid=lc.targetid, label=lc.label)

    def smooth(self, smooth_factor = 10):
        """Smooths the powerspectrum using a moving median filter.

        Parameters
        ----------
        smooth_factor : int
            Default 10. The factor by which to smooth the power spectrum, in the
            sense that the power spectrum will be smoothed by taking the median
            in bins of size N / smooth_factor, where N is the length of the
            original periodogram.

        Returns
        -------
        smooth_periodogram : a `Periodogram` object
            Returns a `Periodogram` object which has been smoothed in bins of
            width `smooth_factor`.
        """
        if smooth_factor < 1:
            raise ValueError('The smooth factor must be greater than 1.')

        #Calculating the length of the smoothed array
        m = int(len(self.power) / smooth_factor)

        smooth_freq = self.frequency[:m*smooth_factor].reshape((m, smooth_factor)).mean(1)
        smooth_power = self.power[:m*smooth_factor].reshape((m, smooth_factor)).mean(1)

        smooth_pg = copy.deepcopy(self)
        smooth_pg.frequency = smooth_freq
        smooth_pg.power = smooth_power
        return smooth_pg

    def plot(self, scale='linear', ax=None, xlabel=None, ylabel=None, title='',
                 style='lightkurve',format='frequency', **kwargs):

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
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        if ylabel is None:
            try:
                ylabel = "Power Spectral Density [ppm$^2\ ${}]".format((1/self.frequency).unit.to_string('latex'))
            except AttributeError:
                pass

        # This will need to be fixed with housekeeping. Self.label currently doesnt exist.
        if ('label' not in kwargs):
            try:
                kwargs['label'] = self.label
            except AttributeError:
                kwargs['label'] = None

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot frequency and power
            if format == 'frequency':
                ax.plot(self.frequency, self.power, **kwargs)
                if xlabel is None:
                    try:
                        xlabel = "Frequency [{}]".format(self.frequency.unit.to_string('latex'))
                    except AttributeError:
                        pass
            if format == 'period':
                ax.plot(self.period, self.power, **kwargs)
                ax.set_xscale('log')
                if xlabel is None:
                    try:
                        xlabel = "Period [{}]".format((1./self.frequency).unit.to_string('latex'))
                    except AttributeError:
                        pass

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            #Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend()
            if scale == "log":
                ax.set_yscale('log')
                ax.set_xscale('log')
            ax.set_title(title)
        return ax

    def estimate_background(self, log_width=0.01):
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

    def remove_background(self, log_width=0.01):
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
        snr_pg = self / self.estimate_background(log_width=log_width)
        return SNR_Periodogram(snr_pg.frequency, snr_pg.power,
                                nyquist = self.nyquist,
                                frequency_spacing = self.frequency_spacing,
                                meta = self.meta)

    ############################### HOUSEKEEPING ###############################

    def to_table(self):
        """Export the Periodogram as an AstroPy Table.
        Returns
        -------
        table : `astropy.table.Table` object
            An AstroPy Table with columns 'frequency', 'period', and 'power'.
        """
        return Table(data=(self.frequency, self.period, self.power),
                     names=('frequency', 'period', 'power'),
                     meta=self.meta)

    def to_pandas(self, columns=['frequency','period','power']):
        """Export the Periodogram as a Pandas DataFrame.
        Parameters
        ----------
        columns : list of str
            List of columns to include in the DataFrame.  The names must match
            attributes of the `Periodogram` object (i.e. `frequency`, `power`)
        Returns
        -------
        dataframe : `pandas.DataFrame` object
            A dataframe indexed by `frequency` and containing the columns `power`
            and `period`.
        """
        try:
            import pandas as pd
        # lightkurve does not require pandas, so check for import success.
        except ImportError:
            raise ImportError("You need to install pandas to use the "
                              "LightCurve.to_pandas() method.")
        data = {}
        for col in columns:
            if hasattr(self, col):
                if isinstance(vars(self)[col], astropy.units.quantity.Quantity):
                    data[col] = vars(self)[col].value
                else:
                    data[col] = vars(self)[col].value
        df = pd.DataFrame(data=data, index=self.frequency, columns=columns)
        df.index.name = 'frequency'
        return df

    def to_csv(self, path_or_buf, **kwargs):
        """Writes the Periodogram to a csv file.
        Parameters
        ----------
        path_or_buf : string or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.
        **kwargs : dict
            Dictionary of arguments to be passed to `pandas.DataFrame.to_csv()`.
        Returns
        -------
        csv : str or None
            Returns a csv-formatted string if `path_or_buf=None`,
            returns None otherwise.
        """
        return self.to_pandas().to_csv(path_or_buf=path_or_buf, **kwargs)

    def to_fits(self, path=None, overwrite=False, **extra_data):
        """Export the Periodogram as an astropy.io.fits object.
        Parameters
        ----------
        path : string, default None
            File path, if None returns an astropy.io.fits object.
        overwrite : bool
            Whether or not to overwrite the file
        extra_data : dict
            Extra keywords or columns to include in the FITS file.
            Arguments of type str, int, float, or bool will be stored as
            keywords in the primary header.
            Arguments of type np.array or list will be stored as columns
            in the first extension.
        Returns
        -------
        hdu : astropy.io.fits
            Returns an astropy.io.fits object if path is None
        """
        # kepler_specific_data = {
        #     'TELESCOP': "KEPLER",
        #     'INSTRUME': "Kepler Photometer",
        #     'OBJECT': '{}'.format(self.targetid),
        #     'KEPLERID': self.targetid,
        #     'CHANNEL': self.channel,
        #     'MISSION': self.mission,
        #     'RA_OBJ': self.ra,
        #     'DEC_OBJ': self.dec,
        #     'EQUINOX': 2000,
        #     'DATE-OBS': Time(self.time[0]+2454833., format=('jd')).isot}
        # for kw in kepler_specific_data:
        #     if ~np.asarray([kw.lower == k.lower() for k in extra_data]).any():
        #         extra_data[kw] = kepler_specific_data[kw]
        # return super(KeplerLightCurve, self).to_fits(path=path,
        #                                              overwrite=overwrite,
        #                                              **extra_data)
        raise NotImplementedError('This should be a function!')

    def __repr__(self):
        return('Periodogram(ID: {})'.format(self.targetid))

    def __getitem__(self, key):
        copy_self = copy.copy(self)
        copy_self.frequency = self.frequency[key]
        copy_self.period = self.period[key]
        copy_self.power = self.power[key]
        return copy_self

    def __add__(self, other):
        copy_self = copy.copy(self)
        copy_self.power = copy_self.power + other
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

    ############################## NOT IMPLEMENTED #############################
    def fit_background(self, numax):
        """
        Function to make a simple fit of power laws to the powerspectrum
        background, and returns the best fit coefficients.
        """
        raise NotImplementedError('This is semi-advanced asteroseismology, but doing this quickly may be valuable to people as a learning tool. Will enquire')

    ############################## WIP, UNTOUCHED ##############################
    ## Lets start with periodogram only, before moving on to the seismo stuff
    ## All the seismo steps will have to be verified one by one
    def estimate_numax(self):
        """Estimates the nu max value based on the periodogram

        find_numax() method first estimates the background trend
        and then smoothes the power spectrum using a gaussian filter.
        Peaks are then determined in the smoothed power spectrum.

        Returns:
        --------
        nu_max : float
            The nu max of self.powers. Nu max is in microhertz
        """

        bkg = self.estimate_background(self.frequency.value, self.powers.value)
        df = self.frequency[1].value - self.frequency[0].value
        smoothed_ps = gaussian_filter(self.powers.value / bkg, 10 / df)
        peak_freqs = self.frequency[self.find_peaks(smoothed_ps)].value
        nu_max = peak_freqs[peak_freqs > 5][0]
        return nu_max

    def estimate_delta_nu(self, numax=None):
        """Estimates delta nu value, or large frequency spacing

        Estimates the delta nu value centered around the numax value given. The autocorrelation
        function will find the distancing between consecutive modesself.

        Parameters
        ----------
        numax : float
            The Nu max value to center our autocorrelation function

        Returns
        -------
        delta_nu : float
            So-called large frequency spacing
        """
        def next_pow_two(n):
            i = 1
            while i < n:
                i = i << 1
            return i

        def acor_function(x):
            x = np.atleast_1d(x)
            n = next_pow_two(len(x))
            f = np.fft.fft(x - np.nanmean(x), n=2*n)
            acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
            acf /= acf[0]
            return acf

        # And the autocorrelation function of a lightly smoothed power spectrum
        bkg = self.estimate_background(self.frequency.value, self.powers.value)
        df = self.frequency[1].value - self.frequency[0].value
        acor = acor_function(gaussian_filter(self.powers.value / bkg, 0.5 / df))
        lags = df*np.arange(len(acor))
        acor = acor[lags < 30]
        lags = lags[lags < 30]

        if numax is None:
            raise ValueError("Must provide a nu max value")
        # Expected delta_nu: Stello et al (2009)
        dnu_expected = 0.263 * numax ** 0.772
        peak_lags = lags[self.find_peaks(acor)]
        delta_nu = peak_lags[np.argmin(np.abs(peak_lags - dnu_expected))]
        return delta_nu

    def find_peaks(self, z):
        """ Finds peak index in an array """
        peak_inds = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
        peak_inds = np.arange(1, len(z)-1)[peak_inds]
        peak_inds = peak_inds[np.argsort(z[peak_inds])][::-1]
        return peak_inds

    def estimate_stellar_parameters(self, nu_max, delta_nu, temp=None):
        """ Estimates stellar parameters.

        Estimates mass, radius, and mean density based on nu max, delta nu, and effective
        temperature values.

        Parameters
        ----------
        nu_max : float
            The nu max of self.powers. Nu max is in microhertz.
        delta_nu : float
            Large frequency spacing in microhertz.
        temp : float
            Effective temperature in Kelvin.

        Returns
        -------
        m : float
            The estimated mass of the target. Mass is in solar units.
        r : float
            The estimated radius of the target. Radius is in solar units.
        rho : float
            The estimated mean density of the target. Rho is in solar units.
        """
        return stellar_params(nu_max, delta_nu, temp)


class SNR_Periodogram(Periodogram):
    """Defines a periodogram with different plotting defaults"""
    def __init__(self, *args, **kwargs):
        super(SNR_Periodogram, self).__init__(*args, **kwargs)

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
#
# def standardize_units(numax, deltanu, temp):
#     """Nondimensionalization units to solar units.
#
#     Parameters
#     ----------
#     numax : float
#         Nu max value in microhertz.
#     deltanu : float
#         Large frequency separation in microhertz.
#     temp : float
#         Effective temperature in Kelvin.
#
#     Returns
#     -------
#     v_max : float
#         Nu max value in solar units.
#     delta_nu : float
#         Delta nu value in solar units.
#     temp_eff : float
#         Effective temperature in solar units.
#     """
#     if numax is None:
#         raise ValueError("No nu max value provided")
#     if deltanu is None:
#         raise ValueError("No delta nu value provided")
#     if temp is None:
#         raise ValueError("An assumed temperature must be given")
#
#     #Standardize nu max, delta nu, and effective temperature
#     v_max = numax / numax_s
#     delta_nu = deltanu / deltanu_s
#     temp_eff = temp / teff_s
#
#     return v_max, delta_nu, temp_eff
#
# def estimate_radius(numax, deltanu, temp_eff=None, scaling_relation=1):
#     """Estimates radius from nu max, delta nu, and effective temperature.
#
#     Uses scaling relations from Belkacem et al. 2011.
#
#     Parameters
#     ----------
#     numax : float
#         Nu max value in microhertz.
#     deltanu : float
#         Large frequency separation in microhertz.
#     temp : float
#         Effective temperature in Kelvin.
#
#     Returns
#     -------
#     radius : float
#         Radius of the target in solar units.
#     """
#     v_max, delta_nu, temp_eff = standardize_units(numax, deltanu, temp_eff)
#     # Scaling relation from Belkacem et al. 2011
#     radius = scaling_relation * v_max * (delta_nu ** -2) * (temp_eff ** .5)
#     return radius
#
# def estimate_mass(numax, deltanu, temp_eff=None, scaling_relation=1):
#     """Estimates mass from nu max, delta nu, and effective temperature.
#
#     Uses scaling relations from Kjeldsen & Bedding 1995.
#
#     Parameters
#     ----------
#     numax : float
#         Nu max value in microhertz.
#     deltanu : float
#         Large frequency separation in microhertz.
#     temp : float
#         Effective temperature in Kelvin.
#
#     Returns
#     -------
#     mass : float
#         mass of the target in solar units.
#     """
#     v_max, delta_nu, temp_eff = standardize_units(numax, deltanu, temp_eff)
#     #Scaling relation from Kjeldsen & Bedding 1995
#     mass = scaling_relation * (v_max ** 3) * (delta_nu ** -4) * (temp_eff ** 1.5)
#     return mass
#
# def estimate_mean_density(mass, radius):
#     """Estimates stellar mean density from the mass and radius.
#
#     Uses scaling relations from Ulrich 1986.
#
#     Parameters
#     ----------
#     mass : float
#         Mass in solar units.
#     radius : float
#         Radius in solar units.
#
#     Returns
#     -------
#     rho : float
#         Stellar mean density in solar units.
#     """
#     #Scaling relation from Ulrich 1986
#     rho = (3.0/(4*np.pi) * (mass / (radius ** 3))) ** .5
#     return np.square(rho)
#
# def stellar_params(numax, deltanu, temp):
#     """Returns radius, mass, and mean density from nu max, delta nu, and effective temperature.
#
#     This is a convenience function that allows users to retrieve all stellar parameters
#     with a single function call.
#
#     Parameters
#     ----------
#     numax : float
#         Nu max value in microhertz.
#     deltanu : float
#         Large frequency separation in microhertz.
#     temp : float
#         Effective temperature in Kelvin.
#
#     Returns
#     -------
#     m : float
#         Mass of the target in solar units.
#     r : float
#         Radius of the target in solar units.
#     rho : float
#         Mean stellar density of the target in solar units.
#     """
#     r = estimate_radius(numax, deltanu, temp)
#     m = estimate_mass(numax, deltanu, temp)
#     rho = estimate_mean_density(m, r)
#     return m, r, rho
