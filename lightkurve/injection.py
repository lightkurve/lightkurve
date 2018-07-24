"""Defines UniformDistribution, GaussianDistribution, TransitModel, and SupernovaModel"""

import numpy as np
import lightkurve
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
    Implements a class for creating a planetary transit model using
    batman - documentation at https://www.cfa.harvard.edu/~lkreidberg/batman/.

    Attributes
    ----------
    signaltype : 'Planet'
        The signal type is a planetary transit.
    multiplicative : True
        A planetary transit is multiplied with a lightcurve; therefore,
        the attribute 'multiplicative' is True
    """

    def __init__(self):
        import batman
        self.signaltype = 'Planet'
        self.multiplicative = True
        self.model = batman.TransitParams()

    def __repr__(self):
        return 'TransitModel(' + str(self.__dict__) + ')'

    def add_planet(self, period, rprs, T0=5, a=15., inc=90., ecc=0., w=90., limb_dark='quadratic', u=[0.1, 0.3]):
        """Adds a planet to TransitModel object.

        Parameters
        ----------
        period : float
            Orbital period of new planet
        rprs : float
            Planet radius/star radius of new planet
        T0 : float
            A transit mid-time
        a : float
            A semimajor axis in stellar radii
        inc : float
            An orbital inclination in degrees
        ecc : float
            An orbital eccentricity
        w : float
            Argument of pariapse in degrees
        u : array-like
            Limb darkening coefficients
        limb_dark : choice of 'nonlinear', 'quadratic', 'exponential', 'logarithmic', 'squareroot',
                    'linear', 'uniform', 'power2'
            Limb darkening model
        """

        params = {'period':period, 'rprs':rprs, 'T0':T0, 'a':a, 'inc':inc, 'ecc':ecc, 'w':w, 'limb_dark':limb_dark, 'u':u}
        for key, val in params.items():
            if isinstance(val, (GaussianDistribution, UniformDistribution)):
                setattr(self, key, val.sample())
            else:
                setattr(self, key, val)

        self.model.t0 = self.T0                      #time of inferior conjunction
        self.model.per = self.period
        self.model.rp = self.rprs                    #planet radius (in units of stellar radii)
        self.model.a = self.a                   #semi-major axis (in units of stellar radii)
        self.model.inc = self.inc                     #orbital inclination (in degrees)
        self.model.ecc = self.ecc                      #eccentricity
        self.model.w = self.w                       #longitude of periastron (in degrees)
        self.model.limb_dark = limb_dark        #limb darkening model
        self.model.u = u      #limb darkening coefficients [u1, u2, u3, u4]

    def evaluate(self, time):
        import batman
        """Creates flux array of lightcurve from model.

        Parameters
        ----------
        time : array-like
            Time array over which to create lightcurve

        Returns
        _______
        transit_flux : array-like
            Flux array of lightcurve

        """

        t = time.astype(np.float)  #times at which to calculate light curve
        m = batman.TransitModel(self.model, t, fac=1.0)    #initializes model
        transit_flux = m.light_curve(self.model)
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
    SyntheticLightCurve class
        Returns a SyntheticLightCurve object possessing a synthetic signal.
    """

    if model.multiplicative is True:
        mergedflux = flux * model.evaluate(time.astype(np.float))
    else:
        mergedflux = flux + model.evaluate(time)
    return lightkurve.lightcurve.SyntheticLightCurve(time, flux=mergedflux, flux_err=flux_err,
                               signaltype=model.signaltype)


def recover(time, flux, flux_err, signal_type, method='optimize', source='hsiao', bandpass='kepler', initial_guess=None,
                                    ndim=None, nwalkers=None, nsteps=None, a=15., inc=90., ecc=0., w=90., limb_dark='quadratic', u=[0.1, 0.3]):
    """Recovers a signal from a lightcurve using optimization.  Coming soon: MCMC
    Right now, we can recover any sncosmo source that
    takes only T0, z, and amplitude.

        ----------
    time : array-like
        Time array
    flux : array-like
        Flux array (with injected or real signal)
    flux_err : array-like
        Flux error array
    signal_type : 'Supernova' or 'Planet'
        Signal to recover
    method: 'optimize' or 'mcmc'
        Fitting method.  If 'mcmc' is chosen, 'ndim', 'nwalkers', and 'nsteps' must be defined
    source : string, default 'hsiao'
        The source of the fitted supernova. Right now, we can only fit
        hsiao models (http://adsabs.harvard.edu/abs/2007ApJ...663.1187H) to supernovae.
    initial_guess : [T0, z, amplitude, background], default None
        Guess vector of parameters.  Planet fitting does not take x0, as a
        Box Least-Squares search (http://adsabs.harvard.edu/abs/2002A%26A...391..369K) is performed.

    Returns
    -------
    result.x : tuple (?)
        Fitted parameters.
    """

    import scipy.optimize as op
    import emcee


    def create_initial_guess():
        if signal_type == "Supernova":
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
            T0, z, amplitude, background = theta
            if (z < 0) or (z > 1) or (T0 < np.min(time)) or (T0 > np.max(time)):
                if method == 'optimize':
                    return -1.e99
                elif method == 'mcmc':
                    return -np.inf
            model = SupernovaModel(T0, z=z, amplitude=amplitude, source=source, bandpass=bandpass)
            model = model.evaluate(time) + background
            inv_sigma2 = 1.0/(flux_err**2)
            chisq = (np.sum((flux-model)**2*inv_sigma2))
            lnlikelihood = -0.5*chisq
            return lnlikelihood

        def lnprior(theta):
            T0, z, amplitude, background = theta
            if (z < 0) or (z > 1):
                if method == 'optimize':
                    return -1.e99
                elif method == 'mcmc':
                    return -np.inf
            return 0.0


        def neg_ln_posterior(theta):
            log_posterior = lnprior(theta) + ln_like(theta)
            return -1 * log_posterior

        if method == 'optimize':
            result = op.minimize(neg_ln_posterior, x0=create_initial_guess())
            return result.x

        elif method == 'mcmc':
            ndim, nwalkers = ndim, nwalkers
            guess=create_initial_guess()
            pos = [[guess[0], guess[1], guess[2], guess[3]] + 1e-7*np.random.randn(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_like)

            x = sampler.run_mcmc(pos, nsteps)

            return sampler, x


    elif signal_type == 'Planet':

        import batman

        #First a BLS search:
        import bls

        u1 = [0.0]*len(time)
        v1 = [0.0]*len(time)
        u1 = np.array(u1)
        v1 = np.array(v1)

#time, flux, u, v, number of freq bins (nf), min freq to test (fmin), freq spacing (df), number of bins (nb), min transit dur (qmi), max transit dur (qma)
        nf = 10000.0
        fmin = .035
        df = 0.001
        nbins = 300
        qmi = 0.001
        qma = 0.3

        results = bls.eebls(time, flux, u1, v1, nf, fmin, df, nbins, qmi, qma)

        bls_period = results[1]
        depth = results[3]
        bls_rprs = np.sqrt(depth)

        midtime = ((float(results[6])-float(results[5])) / 2) + results[5]

        bls_T0 = ((midtime / nbins) * bls_period) + min(time)

        #Then optimization fitting:
        def ln_like(theta):
            period, rprs, T0, inc, ecc = theta

            model = TransitModel()
            model.add_planet(period=period, rprs=rprs, T0=T0, a=a, inc=inc, ecc=ecc, w=w, limb_dark=limb_dark, u=u)
            t = time.astype(np.float)
            flux_model = model.evaluate(t)

            inv_sigma2 = 1.0/(flux_err**2)
            chisq = (np.sum((flux - flux_model)**2 * inv_sigma2))
            lnlikelihood = -0.5*chisq

            return -lnlikelihood

        if method == 'optimize':
            result = op.minimize(ln_like, [bls_period, bls_rprs, bls_T0, 87, 0.0])
        elif method == 'mcmc':
            ndim, nwalkers = ndim, nwalkers
            pos = [[bls_period, bls_rprs, bls_T0] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_like)

            x = sampler.run_mcmc(pos, nsteps)

            return sampler, x

        return result.x

    else:
        print('Signal Type not supported.')
        return

def injrec_test(lc, signal_type, ntests, constr, period=None, rprs=None, T0=None, z=None, amplitude=None):
    """Runs an injection and recovery test for many models and one real lightcurve.
    Right now we can only recover SN that take T0, z, and amplitude.

    Parameters
        ----------
    lc : LightCurve class
        Lightcurve object to inject models into.
    signal_type : "Supernova" or "Planet"
        Type of signal to be injected.
    ntests : int
        Number of injections to be performed.
    constr : float
        parameter constraint to determine a recovered signal (for example. 0.03
        demands that all parameters must be within 3% of the injected value to be
        considered recovered)
    period : Distribution class
        A GaussianDistribution or UniformDistribution object from which to draw
        period values.
    rprs : Distribution class
        A GaussianDistribution or UniformDistribution object from which to draw
        rprs values.
    T0 : Distribution class
        A GaussianDistribution or UniformDistribution object from which to draw
        T0 values.
    z : Distribution class
        A GaussianDistribution or UniformDistribution object from which to draw
        z values.
    amplitude : Distribution class
        A GaussianDistribution or UniformDistribution object from which to draw
        amplitude values.

    Returns
    -------
    fraction : float
        Fraction of lightcurves recovered.

    """
    import lightkurve.injection as inj

    if signal_type == 'Supernova':
        nrecovered = 0

        for i in range(ntests):
            T0_test = T0.sample()
            z_test = z.sample()
            amplitude_test = amplitude.sample()

            model = inj.SupernovaModel(T0=T0_test, z=z_test, amplitude=amplitude_test, source='Hsiao', bandpass='Kepler')
            lcinj = lc.inject(model)

            T0_f, z_f, amplitude_f = lcinj.recover('Supernova')

            if abs(T0_f-T0_test) < constr*T0_test and abs(z_f-z_test) < constr*z_test and abs(amplitude_f-amplitude_test) < constr*amplitude_test:
                nrecovered += 1
                print('Recovered: ' + str(T0_test) + ' ' + str(amplitude_test))

        return (nrecovered / ntests)

    elif signal_type == 'Planet':

        nrecovered = 0

        for i in range(ntests):
            period_test = period.sample()
            rprs_test = rprs.sample()
            T0_test = T0.sample()

            model = inj.TransitModel()
            model.add_planet(period=period_test, rprs=rprs_test, T0=T0_test)
            lcinj = lc.inject(model)

            period_f, rprs_f, T0_f = lcinj.recover('Planet')

            if abs(period_f-period_test) < constr*period_test and abs(rprs_f-rprs_test) < constr*rprs_test:
                nrecovered += 1
                print('Recovered: ' + str(period_test) + ' ' + str(rprs_test))
                print(nrecovered)
        return (float(nrecovered)/ float(ntests))


    else:
        print('Signal type not supported.')
        return






















#hello
