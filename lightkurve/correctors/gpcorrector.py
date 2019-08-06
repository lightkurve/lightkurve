import celerite
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from .corrector import Corrector
from ..utils import validate_method
from .. import log, MPLSTYLE

class GPCorrector(Corrector):
    """
    Accepted kernels are:
    "matern32", "shoterm"
    """
    def __init__(self, lc, kernel="matern32", cadence_mask=None, sigma=5):
        if np.any([~np.isfinite(lc.flux), ~np.isfinite(lc.flux_err)]):
            log.warning("NaNs have been removed from the light curve.")
            self.lc = lc.remove_nans()
        else:
            self.lc = lc

        if cadence_mask is None:
            self.cadence_mask = np.ones(len(self.lc.time), dtype=bool)
        else:
            self.cadence_mask = cadence_mask
        self.diag = np.copy(self.lc.flux_err)
        bad_cadences = np.copy(~self.cadence_mask)
        bad_cadences |= self.lc.flatten(21).remove_outliers(sigma=sigma, return_mask=True)[1]
        print(bad_cadences.sum())
        self.diag[bad_cadences] *= 1e10  # This is faster than masking out flux values

        if isinstance(kernel, celerite.terms.Term):
            self.kernel = kernel
        else:
            kernel_str = validate_method(kernel, ["matern32", "sho"])
            self.kernel = self._build_kernel(kernel_str)

        self.gp = celerite.GP(self.kernel, mean=np.nanmean(lc.flux))
        self.gp.compute(self.lc.time, self.diag)
        self.initial_kernel = self.kernel
        self.optimized = False

    def _build_matern32_kernel(self, matern_bounds=None, jitter_bounds=None):
        log_sigma = np.log(np.nanstd(self.lc.flux))
        log_rho = np.log(self.lc.to_periodogram(minimum_period=0.5, maximum_period=50).period_at_max_power.value)
        log_sigma2 = np.log(np.nanmedian(self.lc.flux_err))

        if matern_bounds is None:
            matern_bounds = {'log_sigma': (-2 + log_sigma, 2 + log_sigma),
                            'log_rho': (np.log(0.5), np.log(50))}
        if jitter_bounds is None:
            jitter_bounds = {'log_sigma':(-2 + log_sigma2, 2 + log_sigma2)}

        kernel = celerite.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
        kernel += celerite.terms.JitterTerm(log_sigma=log_sigma2, bounds=jitter_bounds)
        return kernel

    def _build_sho_kernel(self, sho_bounds=None, jitter_bounds=None):
        log_omega0 = np.log(2*np.pi / self.lc.to_periodogram(minimum_period=0.5, maximum_period=50).period_at_max_power.value)
        log_S0 = np.log(np.nanstd(self.lc.flux)**2)
        log_Q = np.log(10)
        log_sigma = np.log(np.nanmedian(self.lc.flux_err))

        if sho_bounds is None:
            sho_bounds = {'log_S0': (-2 + log_S0, 2 + log_S0),
                          'log_Q': (0.2, 7),
                          'log_w0': (np.log(2*np.pi/150), np.log(2*np.pi/0.1))}
        if jitter_bounds is None:
            jitter_bounds = {'log_sigma':(-2 + log_sigma, 2 + log_sigma)}

        kernel = celerite.terms.SHOTerm(log_omega0=log_omega0, log_S0=log_S0,
                                        log_Q=log_Q, bounds=sho_bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=log_sigma, bounds=jitter_bounds)
        return kernel

    def _build_kernel(self, kernel_str="matern32"):
        """Returns a `celerite.terms.Term` object with reasonable initialization
        values.
        """
        if kernel_str == "matern32":
            kernel = self._build_matern32_kernel()
        elif kernel_str == "sho":
            kernel = self._build_sho_kernel()
        return kernel

    def _neg_log_like(self, params, y):
        self.gp.set_parameter_vector(params)
        return -self.gp.log_likelihood(y)

    def _grad_neg_log_like(self, params, y):
        self.gp.set_parameter_vector(params)
        return -self.gp.grad_log_likelihood(y)[1]

    def optimize(self, method="L-BFGS-B"):
        self.optimized = True
        solution = minimize(self._neg_log_like, self.gp.get_parameter_vector(),
                            method=method, bounds=self.gp.get_parameter_bounds(),
                            jac=self._grad_neg_log_like, args=(self.lc.flux))
        self.gp.set_parameter_vector(solution.x)
        return solution

    def _predict_lightcurves(self):
        gp_flux, gp_flux_var = self.gp.predict(self.lc.flux, self.lc.time)

        corrected_lc = self.lc.copy()
        corrected_lc.flux -= (gp_flux - np.mean(gp_flux))
        corrected_lc.flux_err = np.hypot(corrected_lc.flux_err, np.std(gp_flux_var))

        gp_lc = self.lc.copy()
        gp_lc.flux = gp_flux
        gp_lc.flux_err = np.std(gp_flux_var)
        return {'corrected_lc': corrected_lc,
                'gp_lc': gp_lc}

    def correct(self):
        self.optimize()
        return self._predict_lightcurves()['corrected_lc']

    def diagnose(self, ax=None, fast=False):
        """Show a diagnostic plot of the GP. Returns a matplotlib.pyplot.axes object.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object, default None
            Plot window to use
        kwargs: dict
            Keywords to pass to matplotlib.pyplot

        Returns
        -------
        ax : matplotlib.pyplot.axes object, default None
        """
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            self.lc.errorbar(ax=ax, zorder=1, label='Data', normalize=False)

        # Initial Guess
        if not self.optimized:
            gp = celerite.GP(self.initial_kernel, mean=np.nanmean(self.lc.flux))
            gp.compute(self.lc.time, self.diag)
            if fast:
                mu = gp.predict(self.lc.flux, self.lc.time, return_cov=False, return_var=False)
            else:
                mu, var = gp.predict(self.lc.flux, self.lc.time, return_cov=False, return_var=True)
                ax.fill_between(self.lc.time, mu - var**0.5, mu + var**0.5, alpha=0.3, color='r', label='')
            k = self.initial_kernel.get_parameter_dict()
            label = 'Initial Guess\n' + '\n'.join(['{}: {}'.format(key, np.round(k[key], 3))
                                        for key in k.keys()])
            ax.plot(self.lc.time, mu, c='r', lw=1, zorder=2, label=label)

        if self.optimized:
            if fast:
                mu = self.gp.predict(self.lc.flux, self.lc.time, return_cov=False, return_var=False)
            else:
                mu, var = self.gp.predict(self.lc.flux, self.lc.time, return_cov=False, return_var=True)
                ax.fill_between(self.lc.time, mu - var**0.5, mu + var**0.5, alpha=0.3, color='b', label='')
            k = self.kernel.get_parameter_dict()
            label = 'Optimized\n' + '\n'.join(['{}: {}'.format(key, np.round(k[key], 3))
                                        for key in k.keys()])
            ax.plot(self.lc.time, mu, c='b', lw=1, zorder=2, label=label)

        ax.legend()
        return ax
