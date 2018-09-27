import numpy as np
from astropy import units as u
from astropy.stats import LombScargle
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from astropy.units import cds
import logging

log = logging.getLogger(__name__)

__all__ = ['Periodogram', 'estimate_mass', 'estimate_radius',
            'estimate_mean_density', 'stellar_params', 'standardize_units']

###Add uncertainties <- review these [astero]
numax_s = 3090.0 # Huber et al 2013
deltanu_s = 135.1
teff_s = 5777.0

class Periodogram(object):
    ###Update this [houseekeping]
    """Represents a power spectrum.
    Attributes
    ----------
    frequencies : array-like
        List of frequencies.
    powers : array-like
        Power measurements.
    """
    def __init__(self, frequencies=None, powers=None, nyquist=None, df=None):
        self.frequencies = frequencies
        self.powers = powers
        self.nyquist = nyquist
        self.df = df

    ###Add from .csv or from .fits staticmethods
    @staticmethod
    def from_lightcurve(lc, nterms = 1, normalization = 'psd', nyquist_factor = 1,
                        min_frequency = None, max_frequency = None,
                        min_period = None, max_period = None,
                        frequencies = None, periods = None,
                        samples_per_peak = None, freq_unit = 1/u.day, **kwargs):
        """
        Creates a Periodogram object from a LightCurve instance using
        the Lomb-Scargle method.
        By default, the periodogram will be created for a regular grid of
        frequencies from one frequency separation to the Nyquist frequency,
        where the frequency separation is determined as 1 / the time baseline.

        Alternatively, the user can provide a custom regular grid using the
        `frequencies' parameter or a custom regular grid of periods using the
        `periods' parameter. The min frequency and/or max frequency
        (or max period and/or min period) can be passed to set custom
        limits for the frequency grid.

        The number of samples per peak can be set using the
        samples_per_peak parameter. The parameter nterms controls how many
        Fourier terms are used in the model. Note that many terms could lead to
        spurious peaks. Setting the Nyquist_factor will sample the space beyond
        the Nyquist frequency, which may introduce aliasing.

        The unit parameter allows a request for alternative units in frequency
        space. By default frequency is in (1/day) and power in (ppm^2 * day).
        Asteroseismologists for example may want frequency in (microHz) and
        power in (ppm^2 / microHz), in which case they would pass
        `unit = u.microhertz`.

        # By default, x-axis units will be plotted in frequency. If any period-
        # related parameters are passed (e.g. minimum_period, periods) the x-axis
        # will be plotted in period. Alternatively, x_unit = 'period' will force
        # x-axis units to be in period, and x_unit = 'frequency' will do the same
        # for frequency.

        Caution: this method assumes that the LightCurve's time (lc.time)
        is given in units of days.

        Parameters
        ----------
        lc : LightCurve object
            The LightCurve from which to compute the Periodogram.
        nterms : int
            Default 1. Number of terms to use in the Fourier fit.
        normalization : {'standard', 'model', 'log', 'psd'}
            Default 'psd'. Normalization to use for the periodogram.
        nyquist_factor : int
            Default 1. The multiple of the average Nyquist frequency. Is
            overriden by maximum_frequency (or minimum period).
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
            units of freq_unit.
        periods : array-like
            The regular grid of periods to use (as 1/period). If given a unit,
            it is converted to units of freq_unit. If not, it is assumed to be
            in units of 1/freq_unit.
        samples_per_peak : int
            The approximate number of desired samples across the typical peak.
            Caution: this overrides any given parameters for frequencies or
            period.
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

        #Calculate Nyquist frequency & Frequency Bin Width in terms of days
        nyquist = 0.5 * (1./(np.median(np.diff(lc.time))*u.day))
        df = (1./((np.nanmax(lc.time - lc.time[0]))*u.day))

        #Convert these values to requested frequency unit
        try:
            nyquist = nyquist.to(freq_unit)
            df = df.to(freq_unit)
        except UnitConversionError:
            log.warning('This unit is not compatible with frequency.')

        #Case if no frequency bins passed
        if (frequencies is None):
            if (min_frequency is not None) & (max_frequency is not None) & (max_frequency <= min frequency):
                log.warning('User input max frequency is smaller than or equal to min frequency.\nFrequency range set to defaults.')
                max_frequency = None
                min_frequency = None
            if min_frequency is None:
                min_frequency = df.value
            if max_frequency is None:
                max_frequency = nyquist.value * nyquist_factor

            frequencies = np.arange(min_frequency, max_frequency, df.value)

        ########I need to make a flowchart to figure this out

        #Set in/convert to desired units
        frequencies = u.Quantity(frequencies, freq_unit)

        #######################Apply the LombScargle method
        LS = LombScargle(lc.time * u.day, lc.flux * 1e6,
                            nterms=nterms, normalization=normalization, **kwargs)
        if samples_per_peak is None:
            power = LS.power(frequencies, method="fast")
        elif samples_per_peak is not None:
            power = LS.autopower(samples_per_peak = samples_per_peak,
                                minimum_frequency = min_frequency,
                                maximum_frequency = max_frequency)
        # Ask gully about this on friday

        #Lets normalise the according to parsevals theorem
        norm = np.std(lc.flux * 1e6)**2 / np.sum(power)
        power *= norm

        #Rescale power to units of ppm^2 / days (or other frequency unit)
        power = power / df.value

        ### Periodogram needs properties
        return Periodogram(frequencies=frequencies, powers=power,
                            nyquist=nyquist, df=df)

    def plot(self, scale="linear", ax=None, **kwargs):
        ### Update this so nomenclature is identical to LightCurve.plot
        ### Improve labels
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
        ax.set_xlabel("Frequency [{}]".format(self.df.unit.to_string('latex')))
        ax.set_ylabel("Power [ppm$^2\ ${}]".format((1/self.df).unit.to_string('latex')))


        #Nonfunctional for now, removed
        ### Try out this methodology
        # if numax:
        #     ax.fill_between([numax.value*0.8, numax.value*1.2],
        #                     self.powers.value.min(),
        #                     self.powers.value.max(),
        #                     alpha=0.2, color='C3', zorder=10)

        #I feel like scale should probably be log by default
        if scale == "log":
            ax.set_yscale('log')
            ax.set_xscale('log')
        return ax


    ### Lets start with periodogram only, before moving on to the seismo stuff
    ### All the seismo steps will have to be verified one by one
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
