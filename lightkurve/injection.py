"""Defines UniformDistribution, GaussianDistribution, TransitModel, and SupernovaModel"""

import numpy as np
from lightkurve import LightCurve

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

    #def __repr__():
        #pass

    def sample(self, size=1):
        """Chooses values from uniform distribution.

        Parameters
        ----------
        size : int
            Number of values to return.

        Returns
        -------
        values : array-like
            Returns array of randomly chosen values.
        """
        return np.random.uniform(self.lb, self.ub, size)

    def plot():
        pass

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

    def __repr__():
        pass

    def sample(self, size=1):
        """Chooses values from Gaussian distribution.

        Parameters
        ----------
        size : int
            Number of values to return.

        Returns
        -------
        values : array-like
            Returns array of randomly chosen values.
        """
        return np.random.normal(self.mean, self.var, size)

    def plot():
        pass


class TransitModel(object):
    """
    Implements a class for creating a transiting model using ktransit.

    Attributes
    ----------
    period : float, default chosen from a uniform dist. of 0-20
        Orbital period in days
    rprs : float, default chosen from a uniform dist. of 0-0.4
        Planet radius / star radius
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
    zpt : float, default 1.0
        A photometric zeropoint
    ecosw, esinw : floats, default 0.0
        An eccentricity vector
    occ : float, default 0.0
        a secondary eclipse depth in ppm

    """

    def __init__(self, period=UniformDistribution(0,20).sample(),
                    rprs=UniformDistribution(0,0.4).sample(),
                    impact=UniformDistribution(0,1).sample(),
                    ld1=UniformDistribution(0,1).sample(), ld2=UniformDistribution(0,1).sample(),
                    ld3=0.0, ld4=0.0, dil=0.0, rho=1.5, zpt=1.0,
                    ecosw=0.0, esinw=0.0, occ=0.0):

        self.period = period
        self.rprs = rprs
        self.rho = rho
        self.ld1, self.ld2, self.ld3, self.ld4 = ld1, ld2, ld3, ld4
        self.dil = dil
        self.zpt = zpt
        self.impact = impact
        self.ecosw = ecosw
        self.esinw = esinw
        self.occ = occ
        self.multiplicative = True

    def evaluate(self, time, t0=None):
        """Evaluates synthetic transiting planet light curve from model.
           Currently, we can only create one planet at a time.

        Parameters
        ----------
        time : array-like
            Time array of transit light curve
        t0 : float, default chosen from a uniform distribution of all time values
            Transit mid-time

        Returns
        -------
        transit_flux : array-like
            Returns the flux of the model lightcurve.
        """
        import ktransit

        if t0==None:
            t0=UniformDistribution(time[0],time[-1]).sample()
        else:
            t0=t0

        model = ktransit.LCModel()
        model.add_star(rho=self.rho, ld1=self.ld1, ld2=self.ld2, ld3=self.ld3, ld4=self.ld4, dil=self.dil, zpt=self.zpt)
        model.add_planet(T0=t0, period=self.period, impact=self.impact, rprs=self.rprs,
                         ecosw=self.ecosw, esinw=self.esinw, occ=self.occ)
        model.add_data(time=time)

        transit_flux = model.transitmodel

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
    """
    def __init__(self, source='hsiao', bandpass='kepler', z=0.5):

        self.source = source
        self.bandpass = bandpass
        self.z = z
        self.multiplicative = False

    def evaluate(self, time, t0=None, size=1, **params):
        """Evaluates synthetic supernova light curve from model.
           Currently, we can only create one supernova at a time.

        Parameters
        ----------
        time : array-like
            Time array of supernova light curve
        t0 : float, default chosen from a uniform distribution of all time values
            Time of supernova's beginning or peak brightness, depending on source chosen.
        params : dict
            Dictionary of keyword arguments to be passed to model.set that
            specify the supernova based on the chosen model.

        Returns
        -------
        bandflux : array-like
            Returns the flux of the model lightcurve.
        """

        import sncosmo

        if t0==None:
            t0=UniformDistribution(time[0],time[-1]).sample()
        else:
            t0=t0

        model = sncosmo.Model(source=self.source)
        model.set(t0=t0, z=self.z, **params)
        bandflux = model.bandflux(self.bandpass, time)

        return bandflux

def inject(lc, model, **params):
    """Injects synthetic model into a light curve.

    Parameters
    ----------
    lc : LightCurve object
        Lightcurve to be injected into
    model : SupernovaModel or TransitModel object
         Model lightcurve to be injected
    size : int
        Number of t0 values to be randomly chosen (must always be 1 - change this)
    params : dict
        Dictionary of keyword arguments to be passed to model.evaluate that
        specify the model.

    Returns
    -------
    lc : LightCurve class
        Returns a lightcurve possessing a synthetic signal.
    """

    if model.multiplicative is True:
        mergedflux = lc.flux * model.evaluate(lc.time, **params)
    else:
        mergedflux = lc.flux + model.evaluate(lc.time, **params)
    return LightCurve(lc.time, flux=mergedflux, flux_err=lc.flux_err)
