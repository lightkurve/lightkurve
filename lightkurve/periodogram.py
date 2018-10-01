import numpy as np
from astropy import units as u
from astropy.stats import LombScargle
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from astropy.units import cds
import logging
from . import PACKAGEDIR, MPLSTYLE


log = logging.getLogger(__name__)

__all__ = ['Periodogram', 'estimate_mass', 'estimate_radius',
            'estimate_mean_density', 'stellar_params', 'standardize_units']

###Add uncertainties <- review these [astero]
numax_s = 3090.0 # Huber et al 2013
deltanu_s = 135.1
teff_s = 5777.0

class InputError(Exception):
    """Raised if user inputs both frequency and period kwargs"""
    pass

class Periodogram(object):
    ###Update this [houseekeping]
    """Represents a power spectrum.
    Attributes
    ----------
    frequencies : array-like
        List of frequencies.
    power : array-like
        Power measurements.
    nyquist : float
        The Nyquist frequency of the lightcurve
    frequency_spacing : float
        The frequency spacing of the periodogram.
    """
    def __init__(self, lc=None, frequencies=None, power=None, nyquist=None, fs=None,
                _format = 'frequency'):
        self.lc = lc
        self.frequencies = frequencies
        self.power = power
        self.nyquist = nyquist
        self.frequency_spacing = fs
        self._format = _format

    @staticmethod
    def from_lightcurve(lc, nterms = 1, nyquist_factor = 1, samples_per_peak = 1,
                        min_frequency = None, max_frequency = None,
                        min_period = None, max_period = None,
                        frequencies = None, periods = None,
                        freq_unit = 1/u.day, _format = 'frequency', **kwargs):
        """Creates a Periodogram object from a LightCurve instance using
        the Lomb-Scargle method.
        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        The min frequency and/or max frequency (or max period and/or min period)
        can be passed to set custom limits for the frequency grid. Alternatively,
        the user can provide a custom regular grid using the `frequencies`
        parameter or a custom regular grid of periods using the `periods`
        parameter.

        The the spectrum can be oversampled by increasing the samples_per_peak
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
        samples_per_peak : int
            The approximate number of desired samples across the typical peak.
            This effectively oversamples the spectrum.
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
        frequencies :  array-like
            The regular grid of frequencies to use. If given a unit, it is
            converted to units of freq_unit. If not, it is assumed to be in
            units of freq_unit. This over rides any set frequency limits.
        periods : array-like
            The regular grid of periods to use (as 1/period). If given a unit,
            it is converted to units of freq_unit. If not, it is assumed to be
            in units of 1/freq_unit. This overrides any set period limits.
        freq_unit : `astropy.units.core.CompositeUnit`
            Default: 1/u.day. The desired frequency units for the Lomb Scargle
            periodogram. This implies that 1/freq_unit is the units for period.
        _format : str
            {'frequency', 'period'}. Default 'frequency'. Determines the format
            of the periodogram. If 'frequency', x-axis units will be frequency.
            If 'period', the x-axis units will be period. _format is set
            according to any frequency- or period- related input kwargs, e.g.
            specifying a value of 'max_period' will set _format = 'period'.
        kwargs : dict
            Keyword arguments passed to `astropy.stats.LombScargle()`

        Returns
        -------
        Periodogram : `Periodogram` object
            Returns a Periodogram object extracted from the lightcurve.
        """
        #Check if any values of period have been passed and set format accordingly
        if not all(b is None for b in [periods, min_period, max_period]):
            _format = 'period'

        #Check input consistency
        if (not all(b is None for b in [periods, min_period, max_period])) &\
            (not all(b is None for b in [frequencies, min_frequency, max_frequency])):
            raise InputError('You have input keyword arguments for both frequency and period. Please only use one or the other.')

        #Calculate Nyquist frequency & Frequency Bin Width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(lc.time))*u.day))
        fs = (1./((np.nanmax(lc.time - lc.time[0]))*u.day)) / samples_per_peak

        #Convert these values to requested frequency unit
        nyquist = nyquist.to(freq_unit)
        fs = fs.to(freq_unit)

        #Check if period has been passed
        if periods is not None:
            log.warning('You have passed a grid of periods, which overrides any period/frequency limit kwargs.')
            frequencies = 1./periods
        if max_period is not None:
            min_frequency = 1./max_period
        if min_period is not None:
            max_frequency = 1./min_period

        if frequencies is not None:
            log.warning('You have passed a grid of frequencies, which overrides any period/frequency limit kwargs.')

        #Do unit conversions if user input min/max frequency or period
        elif frequencies is None:
            if min_frequency is not None:
                min_frequency = u.Quantity(min_frequency, freq_unit)
            if max_frequency is not None:
                max_frequency = u.Quantity(max_frequency, freq_unit)
            if (min_frequency is not None) & (max_frequency is not None):
                if (max_frequency <= min_frequency):
                    if _format == 'frequency':
                        raise InputError('User input max frequency is smaller than or equal to min frequency.')
                    if _format == 'period':
                        raise InputError('User input max period is smaller than or equal to min period.')
            #If nothing has been passed in, set them to the defaults
            if min_frequency is None:
                min_frequency = fs
            if max_frequency is None:
                max_frequency = nyquist * nyquist_factor

            #Create frequency grid evenly spaced in frequency
            frequencies = np.arange(min_frequency.value, max_frequency.value, fs.value)

        #Set in/convert to desired units
        frequencies = u.Quantity(frequencies, freq_unit)

        if nterms > 1:
            log.warning('Nterms has been set larger than 1. Method has been set to `fastchi2`')
            method = 'fastchi2'
            if periods is not None:
                method = 'chi2'
                log.warning('You have passed an eventy-spaced grid of periods. These are not evenly spaced in frequency space.\n Method has been set to "chi2" to allow for this.')
        else:
            method='fast'
            if periods is not None:
                method = 'slow'
                log.warning('You have passed an evenly-spaced grid of periods. These are not evenly spaced in frequency space.\n Method has been set to "slow" to allow for this.')


        LS = LombScargle(lc.time * u.day, lc.flux * 1e6,
                            nterms=nterms, normalization='psd', **kwargs)
        power = LS.power(frequencies, method=method)

        #Normalise the according to Parseval's theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        #Rescale power to units of ppm^2 / [frequency unit]
        power = power / fs.value

        ### Periodogram needs properties
        return Periodogram(lc = lc, frequencies=frequencies, power=power,
                            nyquist=nyquist, fs=fs, _format=_format)

    def plot(self, scale='linear', ax=None,
                    xlabel=None, ylabel=None, title='',
                    style='lightkurve',format=None,  **kwargs):

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
            {'frequency', 'period'}. Is by default the _format property of the
            Periodogram object. If 'frequency', x-axis units will be frequency.
            If 'period', the x-axis units will be period and 'log' scale.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if format is None:
            format = self._format
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        if ylabel is None:
            ylabel = "Power Spectral Density [ppm$^2\ ${}]".format((1/self.frequency_spacing).unit.to_string('latex'))

        # This will need to be fixed with housekeeping. Self.label currently doesnt exist.
        if ('label' not in kwargs):
            kwargs['label'] = self.lc.label

        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot frequency and power
            if format == 'frequency':
                ax.plot(self.frequencies, self.power, **kwargs)
                if xlabel is None:
                    xlabel = "Frequency [{}]".format(self.frequency_spacing.unit.to_string('latex'))

            if format == 'period':
                ax.plot(1./self.frequencies, self.power, **kwargs)
                ax.set_xscale('log')
                if xlabel is None:
                    xlabel = "Period [{}]".format((1./self.frequency_spacing).unit.to_string('latex'))

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


################################################################################
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

        bkg = self.estimate_background(self.frequencies.value, self.powers.value)
        df = self.frequencies[1].value - self.frequencies[0].value
        smoothed_ps = gaussian_filter(self.powers.value / bkg, 10 / df)
        peak_freqs = self.frequencies[self.find_peaks(smoothed_ps)].value
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
        bkg = self.estimate_background(self.frequencies.value, self.powers.value)
        df = self.frequencies[1].value - self.frequencies[0].value
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

    def estimate_background(self, x, y, log_width=0.01):
        """Estimates background noise

        Estimates the background noise via median value

        Parameters
        ----------
        self : Periodogram object
            Periodogram object
        x : array-like
            Time measurements
        y : array-like
            Flux measurements
        log_width : float
            The error range

        Returns
        -------
        bkg : float, array-like
            background trend
        """
        count = np.zeros(len(x), dtype=int)
        bkg = np.zeros_like(x)
        x0 = np.log10(x[0])
        while x0 < np.log10(x[-1]):
            m = np.abs(np.log10(x) - x0) < log_width
            bkg[m] += np.nanmedian(y[m])
            count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

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

def standardize_units(numax, deltanu, temp):
    """Nondimensionalization units to solar units.

    Parameters
    ----------
    numax : float
        Nu max value in microhertz.
    deltanu : float
        Large frequency separation in microhertz.
    temp : float
        Effective temperature in Kelvin.

    Returns
    -------
    v_max : float
        Nu max value in solar units.
    delta_nu : float
        Delta nu value in solar units.
    temp_eff : float
        Effective temperature in solar units.
    """
    if numax is None:
        raise ValueError("No nu max value provided")
    if deltanu is None:
        raise ValueError("No delta nu value provided")
    if temp is None:
        raise ValueError("An assumed temperature must be given")

    #Standardize nu max, delta nu, and effective temperature
    v_max = numax / numax_s
    delta_nu = deltanu / deltanu_s
    temp_eff = temp / teff_s

    return v_max, delta_nu, temp_eff

def estimate_radius(numax, deltanu, temp_eff=None, scaling_relation=1):
    """Estimates radius from nu max, delta nu, and effective temperature.

    Uses scaling relations from Belkacem et al. 2011.

    Parameters
    ----------
    numax : float
        Nu max value in microhertz.
    deltanu : float
        Large frequency separation in microhertz.
    temp : float
        Effective temperature in Kelvin.

    Returns
    -------
    radius : float
        Radius of the target in solar units.
    """
    v_max, delta_nu, temp_eff = standardize_units(numax, deltanu, temp_eff)
    # Scaling relation from Belkacem et al. 2011
    radius = scaling_relation * v_max * (delta_nu ** -2) * (temp_eff ** .5)
    return radius

def estimate_mass(numax, deltanu, temp_eff=None, scaling_relation=1):
    """Estimates mass from nu max, delta nu, and effective temperature.

    Uses scaling relations from Kjeldsen & Bedding 1995.

    Parameters
    ----------
    numax : float
        Nu max value in microhertz.
    deltanu : float
        Large frequency separation in microhertz.
    temp : float
        Effective temperature in Kelvin.

    Returns
    -------
    mass : float
        mass of the target in solar units.
    """
    v_max, delta_nu, temp_eff = standardize_units(numax, deltanu, temp_eff)
    #Scaling relation from Kjeldsen & Bedding 1995
    mass = scaling_relation * (v_max ** 3) * (delta_nu ** -4) * (temp_eff ** 1.5)
    return mass

def estimate_mean_density(mass, radius):
    """Estimates stellar mean density from the mass and radius.

    Uses scaling relations from Ulrich 1986.

    Parameters
    ----------
    mass : float
        Mass in solar units.
    radius : float
        Radius in solar units.

    Returns
    -------
    rho : float
        Stellar mean density in solar units.
    """
    #Scaling relation from Ulrich 1986
    rho = (3.0/(4*np.pi) * (mass / (radius ** 3))) ** .5
    return np.square(rho)

def stellar_params(numax, deltanu, temp):
    """Returns radius, mass, and mean density from nu max, delta nu, and effective temperature.

    This is a convenience function that allows users to retrieve all stellar parameters
    with a single function call.

    Parameters
    ----------
    numax : float
        Nu max value in microhertz.
    deltanu : float
        Large frequency separation in microhertz.
    temp : float
        Effective temperature in Kelvin.

    Returns
    -------
    m : float
        Mass of the target in solar units.
    r : float
        Radius of the target in solar units.
    rho : float
        Mean stellar density of the target in solar units.
    """
    r = estimate_radius(numax, deltanu, temp)
    m = estimate_mass(numax, deltanu, temp)
    rho = estimate_mean_density(m, r)
    return m, r, rho
