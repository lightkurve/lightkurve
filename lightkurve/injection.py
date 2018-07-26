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
    'batman' - documentation at https://www.cfa.harvard.edu/~lkreidberg/batman/.

    Attributes
    ----------
    signaltype : 'Planet'
        The signal type is a planetary transit.
    multiplicative : True
        A planetary transit must be multiplied with a lightcurve; therefore,
        the attribute 'multiplicative' is True
    """

    def __init__(self):
        import batman
        self.signaltype = 'Planet'
        self.heritage = 'batman'
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
            Limb darkening model to be used
        """

        self.params = {'period':period, 'rprs':rprs, 'T0':T0, 'a':a, 'inc':inc, 'ecc':ecc, 'w':w, 'limb_dark':limb_dark, 'u':u}
        for key, val in self.params.items():
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
    def __init__(self, source='hsiao', bandpass='kepler', **kwargs):

        self.signaltype = 'Supernova'
        self.heritage = 'sncosmo'
        self.source = source
        #T0 should also have an isinstance Distribution Class if statement
        self.bandpass = bandpass
        self.multiplicative = False

        self.sn_params = {}
        for key, value in kwargs.items():
            if isinstance(value, (GaussianDistribution, UniformDistribution)):
                self.sn_params[key] = value.sample()
            else:
                self.sn_params[key] = value

        if 'T0' in self.sn_params:
            self.sn_params['t0'] = self.sn_params['T0']
            self.sn_params.pop('T0', None)

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
        model.set(**self.sn_params)
        band_intensity = model.bandflux(self.bandpass, time)  #Units: e/s/cm^2
        if self.bandpass == 'kepler':
            # Spectral response already includes QE: units are e- not photons.
            # see Kepler Instrument Handbook
            A_eff_cm2 = 5480.0 # Units cm^2
            bandflux = band_intensity * A_eff_cm2  #e/s
        else:
            bandflux = band_intensity # Units: photons/s/cm^2

        self.params = self.sn_params.copy()
        signal_dict = {'signal':bandflux, 'bandpass':self.bandpass}
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
                               **model.params)

"""
def recover_planet(time, flux, flux_err, method='optimize')
#synlc.recover_planet(method="optimize"...)
#IN RECOVER METHOD IN SyntheticLightCurve:
    self.period
    self.rprs

    for param:
        if not in list passed in:
            create TransitModel(....inc=self.inc)
            try it on, and then minimize
            .....






    Which params do we want to fit? (for example...)
    ['period', 'rprs', 'T0']
    We could pass in a list of strings - the names of the params we want to fit.

    Which params do we want to fix?
    Everything besides what is in the list above.





_____________

def recover(time, flux, .... , period, rprs, inc...... ['period', 'rprs'])





IN SyntheticLightCurve:
import injection as inj
    def recover(fit_params = ['period', 'rprs'])
        inj.recover()

"""


def recover_planet(time, flux, flux_err, period, rprs, T0, a, inc, ecc, w, limb_dark, u,
                    fit_params, method='optimize', nwalkers=10, nsteps=100, threads=1):

    import scipy.optimize as op
    import emcee
    import batman
    import bls

    def create_initial_guess(fit_params=fit_params):
        u1 = np.array([0.0]*len(time))
        v1 = np.array([0.0]*len(time))

        #time, flux, u, v, number of freq bins (nf), min freq to test (fmin), freq spacing (df), number of bins (nb), min transit dur (qmi), max transit dur (qma)
        nf = 10000.0
        #we can only fit periods up to 33.3 or the jupyter notebook kernel dies? hmm
        fmin = .03
        df = 0.001
        nbins = 300
        qmi = 0.001
        qma = 0.3

        results = bls.eebls(time, flux, u1, v1, nf, fmin, df, nbins, qmi, qma)

        bls_period = results[1]
        bls_rprs = np.sqrt(results[3])

        midtime = ((float(results[6])-float(results[5])) / 2) + results[5]
        bls_T0 = ((midtime / nbins) * bls_period) + min(time)

        guess = {}
        if 'period' in fit_params:
            guess['period'] = bls_period
        if 'rprs' in fit_params:
            guess['rprs'] = bls_rprs
        if 'T0' in fit_params:
            guess['T0'] = bls_T0
        if 'a' in fit_params:
            guess['a'] = 15.
        if 'inc' in fit_params:
            guess['inc'] = 90.
        if 'ecc' in fit_params:
            guess['ecc'] = 0.
        if 'w' in fit_params:
            guess['w'] = 90.

        return guess


    def ln_like(theta):
        dict = {'period':period, 'rprs':rprs, 'T0':T0, 'a':a, 'inc':inc, 'ecc':ecc, 'w':w}
        for i in range(len(theta)):
            dict[create_initial_guess().keys()[i]] = theta[i]

        if (dict['rprs'] < 0):
            if method == 'optimize':
                return -1.e99
            elif method == 'mcmc':
                return -np.inf

        print(dict)
        model = TransitModel()
        model.add_planet(period=dict['period'], rprs=dict['rprs'], T0=dict['T0'], a=dict['a'], inc=dict['inc'], ecc=dict['ecc'], w=dict['w'], limb_dark=limb_dark, u=u)
        #model.add_planet(period=period, rprs=rprs, T0=T0, a=a, inc=inc, ecc=ecc, w=w, limb_dark=limb_dark, u=u)

        t = time.astype(np.float)
        model = model.evaluate(t)
        inv_sigma2 = 1.0/(flux_err**2)
        chisq = (np.sum((flux-model)**2*inv_sigma2))
        lnlikelihood = -0.5*chisq
        return lnlikelihood

    def neg_ln_posterior(theta):
        log_posterior = ln_like(theta)
        return -1 * log_posterior

    if method == 'optimize':
        x0 = create_initial_guess().values()
        result = op.minimize(neg_ln_posterior, x0)

        results = dict(zip(create_initial_guess().keys(), result.x))

        return results

    elif method == 'mcmc':
        ndim = len(fit_params)
        pos = [create_initial_guess().values() + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_like, threads=threads)
        x = sampler.run_mcmc(pos, nsteps)
        return sampler


def recover_supernova(time, flux, flux_err, T0, z, amplitude, fit_params, source='hsiao', bandpass='kepler', initial_guess=None,
                     method='optimize', nwalkers=10, nsteps=100, threads=1):

    import scipy.optimize as op
    import emcee

    def create_initial_guess(fit_params=fit_params):
        if initial_guess is None:
            T0 = np.median(time)
            z = 0.45
            amplitude = 6.e-8#is there a better way to guess this?
            background = np.percentile(flux, 3)

            guess = {}
            if 'T0' in fit_params:
                guess['T0'] = T0
            if 'z' in fit_params:
                guess['z'] = z
            if 'amplitude' in fit_params:
                guess['amplitude'] = amplitude
            if 'background' in fit_params:
                guess['background'] = background

            return guess

        else:
            return initial_guess


    def ln_like(theta):
        dict = {}
        dict['T0'] = T0
        dict['z'] = z
        dict['amplitude'] = amplitude
        dict['background'] = 7800
        for i in range(len(theta)):
            dict[create_initial_guess().keys()[i]] = theta[i]
        print(dict)
        if (dict['z'] < 0) or (dict['z'] > 1.5) or (dict['T0'] < np.min(time)) or (dict['T0'] > np.max(time)):
            if method == 'optimize':
                return -1.e99
            elif method == 'mcmc':
                return -np.inf
        model = SupernovaModel(T0=dict['T0'], source=source, bandpass=bandpass, z=dict['z'], amplitude=dict['amplitude'])
        model = model.evaluate(time) + dict['background']
        inv_sigma2 = 1.0/(flux_err**2)
        chisq = (np.sum((flux-model)**2*inv_sigma2))
        lnlikelihood = -0.5*chisq
        return lnlikelihood



    def neg_ln_posterior(theta):
        log_posterior = ln_like(theta)
        return -1 * log_posterior

    if method == 'optimize':
        x0 = create_initial_guess().values()
        result = op.minimize(neg_ln_posterior, x0)

        results = dict(zip(create_initial_guess().keys(), result.x))

        return results

    elif method == 'mcmc':
        ndim = len(fit_params)
        pos = [create_initial_guess().values() + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_like, threads=threads)
        x = sampler.run_mcmc(pos, nsteps)
        return sampler



def injection_and_recovery(lc, signal_type, ntests, constr, period=None, rprs=None, T0=None, inc=None, z=None, amplitude=None,
                        ecc=0.0, a=15., w=90., limb_dark='quadratic', u=[0.1, 0.3]):
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
                print('Recovered - T0: ' + str(T0_test) + ' amplitude: ' + str(amplitude_test) + ' z: ' + str(z_test))

        return (float(nrecovered)/ float(ntests))

    elif signal_type == 'Planet':

        nrecovered = 0

        for i in range(ntests):
            period_test = period.sample()
            rprs_test = rprs.sample()
            T0_test = T0.sample()
            inc_test = inc.sample()

            period_rec = []
            rprs_rec = []
            T0_rec = []
            inc_rec = []

            model = inj.TransitModel()
            model.add_planet(period=period_test, rprs=rprs_test, T0=T0_test, inc=inc_test, ecc=ecc, a=a, w=w, limb_dark=limb_dark, u=u)
            lcinj = lc.inject(model)

            period_f, rprs_f, T0_f, inc_f = lcinj.recover('Planet')

            if abs(period_f-period_test) < constr*period_test and abs(rprs_f-rprs_test) < constr*rprs_test:
                nrecovered += 1
                print('Recovered - period: ' + str(period_test) + ' rprs: ' + str(rprs_test) + ' T0: ' + str(T0_test) + ' inclination: ' + str(inc_test))

        return (float(nrecovered)/ float(ntests))


    else:
        print('Signal type not supported.')
        return






















#hello
