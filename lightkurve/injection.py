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

def inject(time, flux, flux_err, model):
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
        mergedflux = flux * model.evaluate(time)
    else:
        mergedflux = flux + model.evaluate(time)
    return lightkurve.lightcurve.SyntheticLightCurve(time, flux=mergedflux, flux_err=flux_err,
                               signaltype=model.signaltype, **model.params)


def recover(time, flux, flux_err, signal_type, source='hsiao', bandpass='kepler', initial_guess=None):
    """Recovers a signal from a lightcurve using optimization.  Coming soon: MCMC
    Right now, we can only recover SALT1 and SALT2 supernovae, and any source that
    takes only T0, z, and amplitude (which are most of them). I'm not sure if there are any
    others but I'll have to check.

    Parameters
        ----------
    time : array-like
        Time array
    flux : array-like
        Flux array (with injected or real signal)
    flux_err : array-like
        Flux error array
    signal_type : 'Supernova' or 'Planet'
         Signal to recover
    source : str (one of built in sources in sncosmo), default None
        The source of the fitted supernova.
    initial_guess : [T0, z, amplitude, background] or [T0, z, x0, x1, c], default None
        Guess vector of parameters. Supernovae must take x0,
        and Planets do not take x0.

    Returns
    -------
    results.x : tuple
        Fitted parameters.
    """

    import scipy.optimize as op


    def create_initial_guess():
        if signal_type == "Supernova":
            if source == 'SALT1' or source == 'SALT2':
                return initial_guess
            else:
                if initial_guess is None:
                    T0 = np.median(time)
                    z = 0.5
                    amplitude = 3.e-7
                    background = np.percentile(flux, 3)
                    return [T0, z, amplitude, background]
                else:
                    return initial_guess

    if signal_type == 'Supernova':

        def ln_like(theta):
            if source == 'SALT1' or source == 'SALT2':
                T0, z, x0, x1, c, background = theta
                #this makes no sense lol -- TODO: FIND OUT WHAT SALT1 AND SALT2 VALS VIOLATE THE BANDPASS
                if (z < 0) or (z > 3) or (T0 < np.min(time)) or (T0 > np.max(time)) or (x0 < 0) or (x0 > 1) or (x1 < 0) or (x1 > 1) or (c < -0.5) or (c > 0.5):
                    return -1.e99
                model = SupernovaModel(T0, z=z, x0=x0, x1=x1, c=c, source=source, bandpass=bandpass)
            else:
                T0, z, amplitude, background = theta
                if (z < 0) or (z > 3) or (T0 < np.min(time)) or (T0 > np.max(time)):
                    return -1.e99
                model = SupernovaModel(T0, z=z, amplitude=amplitude, source=source, bandpass=bandpass)

            model = model.evaluate(time) + background
            inv_sigma2 = 1.0/(flux_err**2)
            chisq = (np.sum((flux-model)**2*inv_sigma2))
            lnlikelihood = -0.5*chisq
            return lnlikelihood


        def lnprior_optimization(theta):
            if source == 'SALT1' or source == 'SALT2':
                T0, z, x0, x1, c, background = theta
            else:
                T0, z, amplitude, background = theta
            if (z < 0) or (z > 3):
                return -1.e99
            return 0.0


        def neg_ln_posterior(theta):
            log_posterior = lnprior_optimization(theta) + ln_like(theta)
            return -1 * log_posterior

        result = op.minimize(neg_ln_posterior, x0=create_initial_guess())

        return result.x


    elif signal_type == 'Planet':

        import batman

        #First a BLS search:
        from astropy.stats import BLS

        model = BLS(time, flux, dy=0.01)
        periodogram = model.autopower(0.2)
        best_index = np.argmax(periodogram.power)
        bls_period = periodogram.period[best_index]
        depth = periodogram.depth[best_index]
        bls_rprs = np.sqrt(depth)

        bls_T0 = periodogram.transit_time[best_index]

        def ln_like(theta):
            period, rprs, T0 = theta

            params = batman.TransitParams()       #object to store transit parameters
            params.t0 = T0                        #time of inferior conjunction
            params.per = period                   #orbital period
            params.rp = rprs                      #planet radius (in units of stellar radii)
            params.a = 15.                        #semi-major axis (in units of stellar radii)
            params.inc = 90.                      #orbital inclination (in degrees)
            params.ecc = 0.                       #eccentricity
            params.w = 90.                        #longitude of periastron (in degrees)
            params.limb_dark = "nonlinear"        #limb darkening model
            params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]

            t = time.astype(np.float)
            m = batman.TransitModel(params, t, fac=1.0)
            flux_model = m.light_curve(params)

            inv_sigma2 = 1.0/(flux_err**2)
            chisq = (np.sum((flux - flux_model)**2 * inv_sigma2))
            lnlikelihood = -0.5*chisq

            return -lnlikelihood

        result = op.minimize(ln_like, [bls_period, bls_rprs, bls_T0])

        return result.x

    else:
        print('Signal Type not supported')
        return
