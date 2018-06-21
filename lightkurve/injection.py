"""Defines UniformDistribution, GaussianDistribution, TransitModel, and SupernovaModel"""

import numpy as np
from lightkurve import LightCurve
import matplotlib.pyplot as plt
from scipy.stats import norm

class UniformDistribution(object):
    """
    Implements a class for choosing a value from a uniform distribution.

    Attributes
    ----------
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
        """Chooses values from uniform distribution.

        Parameters
        ----------
        None

        Returns
        -------
        values : array-like
            Returns array of randomly chosen values.
        """
        return np.random.uniform(self.lb, self.ub)

    def plot(self):
        """Plots specified distribution."""
        t = np.arange(self.lb, self.ub, 0.01)
        vals = [1]*len(t)
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
        return 'GaussianDistribution(mean={},var={})'.format(self.mean,
                                         self.var)

    def sample(self):
        """Chooses values from Gaussian distribution.

        Parameters
        ----------
        None

        Returns
        -------
        values : array-like
            Returns array of randomly chosen values.
        """
        return np.random.normal(self.mean, self.var)

    def plot(self):
        """Plots specified distribution."""
        t = np.linspace(self.mean - 3*self.var, self.mean + 3*self.var, 100)
        vals = norm.pdf(t, self.mean, self.var)
        plt.plot(t, vals)
        plt.ylim(0, np.max(vals))


class TransitModel(object):
    """
    Implements a class for creating a transiting model using ktransit.

    Attributes
    ----------
    period : float, default chosen from a uniform dist. of 0-20
        Orbital period in days
    rprs : float, default chosen from a uniform dist. of 0-0.4
        Planet radius / star radius
    zpt : float, default 1.0
        A photometric zeropointf
    **kwargs : dict
        Keyword arguments to be passed to model.add_star and
        model.add_planet that specify the transits. Options are:
            T0 : float
                a transit mid-time (note that the T is uppercase)
            impact : float, default chosen from a uniform dist. of 0-1
                Impact parameter
            ld1 : float, default chosen from a uniform dist. of 0-1
                Limb darkening coefficient 1
            ld2 : float, default chosen from a uniform dist. of 0-1
                Limb darkening coefficient 2
            ld3 : float, default 0.0
                Limb darkening coefficient 3
            ld4 : float, default 0.0
                Limb darkening coefficient 4
            dil : float, default 0.0
                Transit dilution fraction
            rho : float, default 1.5
                Mean stellar density in cgs units
            ecosw, esinw : floats, default 0.0
                An eccentricity vector
            occ : float, default 0.0
                a secondary eclipse depth in ppm

    """

    def __init__(self, zpt = 1.0, rho=1.5, ld1 = 0.2, ld2=0.4, ld3=0.0, ld4=0.0, dil=0.0):
        """Initialize the star."""

        """
        self.planet_num = 0
        self.T0 = []
        self.period = []
        self.impact = []
        self.rprs = []
        self.ecosw = []
        self.esinw = []
        self.occ = []
        self.ell = []
        self.alb = []
        """

        self.rho = rho
        self.ld1 = ld1
        self.ld2 = ld2
        self.ld3 = ld3
        self.ld4 = ld4
        self.dil = dil
        self.zpt = zpt

        self.model = ktransit.LCModel()
        self.model.add_star(zpt=self.zpt, rho=self.rho, ld1=self.ld1, ld2=self.ld2, ld3=self.ld3,
                            ld4=self.ld4, dil=self.dil)

        self.multiplicative = True

    def __repr__(self):
        return 'TransitModel(' + str(self.__dict__) + ')'

    def add_planet(self, period, rprs, T0, impact=0.0, ecosw=0.0, esinw=0.0,
                    rvamp = 0.0, occ=0.0, alb=0.0):
        """Modifies existing TransitModel object by adding another planet.

        Parameters
        ----------
        Default values are those initialized in TransitModel.
        A parameter must be defined in the initialization of
        TransitModel if it is to be changed in add_planet.

        period : float
            Orbital period of new planet
        rprs : float
            Planet radius/star radius of new planet
        **added_planet_params : dict
            Dictonary of planet parameters (options the same as in TransitModel)
        """



        """
        self.T0 = T0
        self.period = period
        self.impact = impact
        self.rprs = rprs
        self.ecosw = ecosw
        self.esinw = esinw
        self.occ = occ
        self.alb = alb
        self.planet_num += 1
        self.add_planet_attributes(self.period, self.rprs, self.T0, self.impact,
                                        self.ecosw, self.esinw,
                                        self.occ, self.alb)
        self.period[self.planet_num-1] = period
        self.impact[self.planet_num-1] = impact
        self.rprs[self.planet_num-1] = rprs
        self.ecosw[self.planet_num-1] = ecosw
        self.esinw[self.planet_num-1] = esinw
        self.occ[self.planet_num-1] = occ
        self.alb[self.planet_num-1] = alb
        """

    def add_planet_attributes(self, period, rprs, T0, impact=0.0, ecosw=0.0,
                                esinw=0.0, occ=0.0, alb=0.0):
        """Helper class for add_planet."""
        self.period = np.r_[self.period, 0.0]
        self.rprs = np.r_[self.rprs, 0.0]
        self.T0 = np.r_[self.T0, 0.0]
        self.impact = np.r_[self.impact, 0.0]
        self.ecosw = np.r_[self.ecosw, 0.0]
        self.esinw = np.r_[self.esinw, 0.0]
        self.occ = np.r_[self.occ, 0.0]
        self.alb = np.r_[self.alb, 0.0]

    def evaluate(self, time):
        """Evaluates synthetic transiting planet light curve from model.
           Currently, we can only create one planet at a time.

        Parameters
        ----------
        time : array-like
            Time array of transit light curve

        Returns
        -------
        transit_flux : array-like
            Returns the flux of the model lightcurve.
        """
        import ktransit

        model = ktransit.LCModel()
        model.add_star(zpt=self.zpt, **self.star_params)
        params_to_add = {}
        for i in range(len(self.period)):
            for key, value in self.planet_params.items():
                if len(value) > 1:
                    params_to_add[key] = value[i]
                else:
                    params_to_add[key] = value[0]
            model.add_planet(period=self.period[i], rprs=self.rprs[i], **params_to_add)
        model.add_data(time=time)

        transit_flux = model.transitmodel

        return transit_flux

class SupernovaModel(object):

    from lightkurve.injection import GaussianDistribution
    from lightkurve.injection import UniformDistribution
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
    """
    def __init__(self, T0, source='hsiao', bandpass='kepler', z=0.5, **kwargs):

        self.source = source
        self.T0 = T0
        self.bandpass = bandpass
        if isinstance(z, (GaussianDistribution, UniformDistribution)):
            self.z = z.sample()
        else:
            self.z = z
        self.multiplicative = False

        self.params = {}
        for key, value in kwargs.items():
            if isinstance(value, (GaussianDistribution, UniformDistribution)):
                self.params[key] = value.sample()
            else:
                self.params[key] = value

    def __repr__(self):
        return 'SupernovaModel(' + str(self.__dict__) + ')'

    def evaluate(self, time, size=1):
        """Evaluates synthetic supernova light curve from model.
           Currently, we can only create one supernova at a time.

        Parameters
        ----------
        time : array-like
            Time array of supernova light curve.
        params : dict
            Dictionary of keyword arguments to be passed to model.set that
            specify the supernova based on the chosen model.

        Returns
        -------
        bandflux : array-like
            Returns the flux of the model lightcurve.
        """

        import sncosmo

        model = sncosmo.Model(source=self.source)
        model.set(t0=self.T0, z=self.z, **self.params)
        bandflux = model.bandflux(self.bandpass, time)

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
    return LightCurve(lc.time, flux=mergedflux, flux_err=lc.flux_err)
