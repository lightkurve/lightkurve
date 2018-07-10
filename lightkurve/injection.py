"""Defines UniformDistribution, GaussianDistribution, TransitModel, and SupernovaModel"""

import numpy as np
import lightkurve
import matplotlib.pyplot as plt
from scipy.stats import norm

class UniformDistribution(object):
    """
    Implements a class for choosing a value from a uniform distribution.

    Attributes
    ----------s
    lb : float
        Lower bound of distribution
    ub : float
        Upper bound of distribution
    """
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return 'UniformDistribution(lb={},ub={})'.format(self.lb,
                                         self.ub)

    def sample(self):
        """Returns a value from a uniform distribution."""
        return np.random.uniform(self.lb, self.ub)

    def plot(self):
        """Plots specified distribution."""
        t = np.linspace(self.lb*0.9, self.ub*1.1, 1000)
        vals = [1] * len(t)
        plt.plot(t, vals)
        plt.ylim(0, 2)

class GaussianDistribution(object):
    """
    Implements a class for choosing a value from a Gaussian distribution.

    Attributes
    ----------
    mean : float
        Mean of distribution
    var : float
        Standard deviation of distribution
    """
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __repr__(self):
        return 'GaussianDistribution(mean={}, var={})'.format(self.mean, self.var)

    def sample(self):
        """Returns a value from a Gaussian distribution."""
        return np.random.normal(self.mean, self.var)

    def plot(self):
        """Plots specified distribution."""
        t = np.linspace(self.mean - 3*self.var, self.mean + 3*self.var, 100)
        vals = norm.pdf(t, self.mean, self.var)
        plt.plot(t, vals)
        plt.ylim(0, np.max(vals))


class TransitModel(object):
    """
    Implements a class for creating a planetary transit model using ktransit.

    Attributes
    ----------
    zpt : float
        A photometric zeropoint
    **kwargs : dict
        Star parameters

    """

    def __init__(self):
        import ktransit
        self.signaltype = 'Planet'
        self.multiplicative = True
        self.model = ktransit.LCModel()

    def __repr__(self):
        return 'TransitModel(' + str(self.__dict__) + ')'

    def add_star(self, zpt=1.0, **kwargs):
        """Initializes the star.

        Parameters
        ----------
        Default values are those initialized in TransitModel.

        zpt : float
            A photometric zeropoint
        **kwargs : dict
            Dictonary of planet parameters. Options are:
                rho : stellar density
                ld1, ld2, ld3, ld3 : limb darkening coefficients
                dil : transit dilution fraction
                veloffset : (not sure what this is)
        """

        self.zpt = zpt

        self.star_params = {}
        for key, value in kwargs.items():
            if isinstance(value, (GaussianDistribution, UniformDistribution)):
                self.star_params[key] = value.sample()
            else:
                self.star_params[key] = value

        self.model.add_star(zpt=self.zpt, **self.star_params)

    def add_planet(self, period, rprs, T0, **kwargs):
        """Adds a planet to TransitModel object.

        Parameters
        ----------
        period : float
            Orbital period of new planet
        rprs : float
            Planet radius/star radius of new planet
        T0 : float
            A transit mid-time
        **kwargs : dict
            Dictonary of planet parameters. Options are:
                impact: an impact parameter
                ecosw, esinw : an eccentricity vector
                occ : a secondary eclipse depth
                rvamp : (not sure)
                ell : (not sure)
                alb : (not sure)
        """

        params = {'period':period, 'rprs':rprs, 'T0':T0}
        for key, val in params.items():
            if isinstance(val, (GaussianDistribution, UniformDistribution)):
                setattr(self, key, val.sample())
            else:
                setattr(self, key, val)

        self.planet_params = {}
        for key, value in kwargs.items():
            if isinstance(value, (GaussianDistribution, UniformDistribution)):
                self.planet_params[key] = value.sample()
            else:
                self.planet_params[key] = value

        self.params = self.star_params.copy()
        self.params.update(self.planet_params)
        required_params = {'period':self.period, 'rprs':self.rprs, 'T0':self.T0}
        self.params.update(required_params)

        self.model.add_planet(period=self.period, rprs=self.rprs, T0=self.T0, **self.planet_params)

    def evaluate(self, time):
        """Creates lightcurve from model.

        Parameters
        ----------
        time : array-like
            Time array over which to create lightcurve

        Returns
        _______
        transit_flux : array-like
            Flux array of lightcurve

        """
        self.model.add_data(time=time)
        transit_flux = self.model.transitmodel
        transit_flux_dict = {'signal':transit_flux}
        self.params.update(transit_flux_dict)
        return transit_flux

class SupernovaModel(object):
    """
    Implements a class for creating a supernova model using sncosmo.

    Attributes
    ----------
    source : string, default 'hsiao'
        Source of supernova model (built into sncosmo)
    bandpass : string, default 'kepler'
        Bandpass for supernova signal.  Built-in bandpasses here:
        https://sncosmo.readthedocs.io/en/v1.6.x/bandpass-list.html
    z : float
        Redshift of supernova
    T0 : float, default chosen from a uniform distribution of all time values
        Time of supernova's beginning or peak brightness, depending on source chosen
    **kwargs : dicts
        List of parameters depending on chosen source.
    """
    def __init__(self, T0, source='hsiao', bandpass='kepler', z=0.5, **kwargs):

        self.signaltype = 'Supernova'
        self.source = source
        self.T0 = T0
        self.bandpass = bandpass
        if isinstance(z, (GaussianDistribution, UniformDistribution)):
            self.z = z.sample()
        else:
            self.z = z
        self.multiplicative = False

        self.sn_params = {}
        for key, value in kwargs.items():
            if isinstance(value, (GaussianDistribution, UniformDistribution)):
                self.sn_params[key] = value.sample()
            else:
                self.sn_params[key] = value

    def __repr__(self):
        return 'SupernovaModel(' + str(self.__dict__) + ')'

    def evaluate(self, time):
        """Evaluates synthetic supernova light curve from model.
           Currently, we can only create one supernova at a time.

        Parameters
        ----------
        time : array-like
            Time array of supernova light curve.
        params : dict
            Dictionary of keyword arguments to be passed to sncosmo's
            model.set (in this method) that specify the
            supernova properties based on the chosen source.

        Returns
        -------
        bandflux : array-like
            Returns the flux of the model lightcurve.
        """

        import sncosmo

        model = sncosmo.Model(source=self.source)
        model.set(t0=self.T0, z=self.z, **self.sn_params)
        band_intensity = model.bandflux(self.bandpass, time)  #Units: e/s/cm^2
        if self.bandpass == 'kepler':
            # Spectral response already includes QE: units are e- not photons.
            # see Kepler Instrument Handbook
            A_eff_cm2 = 5480.0 # Units cm^2
            bandflux = band_intensity * A_eff_cm2  #e/s
        else:
            bandflux = band_intensity # Units: photons/s/cm^2
        self.params = self.sn_params.copy()
        signal_dict = {'signal':bandflux}
        self.params.update(signal_dict)

        return bandflux

def inject(lc, model):
    """Injects synthetic model into a light curve.

    Parameters
    ----------
    lc : LightCurve object
        Lightcurve to be injected into
    model : SupernovaModel or TransitModel object
         Model lightcurve to be injected

    Returns
    -------
    lc : LightCurve class
        Returns a lightcurve possessing a synthetic signal.
    """

    if model.multiplicative is True:
        mergedflux = lc.flux * model.evaluate(lc.time)
    else:
        mergedflux = lc.flux + model.evaluate(lc.time)
    return lightkurve.lightcurve.SyntheticLightCurve(lc.time, flux=mergedflux, flux_err=lc.flux_err,
                               signaltype=model.signaltype, **model.params)


def recover(lc, signal_type, x0, **kwargs):
    import scipy.optimize as op

    if signal_type == 'Supernova':

        def ln_like(theta):
            T0, z, amplitude, background = theta
            if (z < 0) or (z > 1) or (T0 < np.min(lc.time)) or (T0 > np.max(lc.time)):
                return -1.e99
            model = SupernovaModel(T0, z=z, amplitude=amplitude, bandpass='kepler')
            model = model.evaluate(lc.time) + background
            inv_sigma2 = 1.0/(lc.flux_err**2)
            chisq = (np.sum((lc.flux-model)**2*inv_sigma2))
            lnlikelihood = -0.5*chisq
            return lnlikelihood


        def lnprior_optimization(theta):
            T0, z, amplitude, background = theta
            if (z < 0) or (z > 1):
                return -1.e99
            return 0.0


        def neg_ln_posterior(theta):
            log_posterior = lnprior_optimization(theta) + ln_like(theta)
            return -1 * log_posterior


        x0 = x0

        result = op.minimize(neg_ln_posterior, x0)

        return result



    elif signal_type == 'Planet':
        pass

    else:
        pass
