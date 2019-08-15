"""Defines the `GPCorrector` for Gaussian Process fitting and removal.
"""
import celerite
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from .corrector import Corrector
from ..utils import validate_method
from .. import log, MPLSTYLE

from .. import log

class GPCorrector(Corrector):
    r"""An object to fit a Gaussian Process (GP) to trends in light curves using
    the celerite implimentation [1]_.

    Accepted values for kernels are:
     - `"matern32"`: fits a Matérn-3/2 kernel
     - `"sho"`: fits a Simple Harmonic Oscillator kernel

    Alternately, a `celerite.terms.Term` object can be passed in a custom kernel.
    For more information about constructing the kernel, see celerite's
    documentation.

    References
    ----------
    .. [1] Celerite documentation, https://celerite.readthedocs.io/en/stable/

    Parameters
    ----------
    lc : Lightkurve.LightCurve object
        The input `~lightkurve.LightCurve` object
    kernel : "matern32" or "sho" or celerite.terms.Term object
        Kernel object to be fit to the light curve. Can be passed in as a string
        ("matern32" for a Matérn-3/2 or "sho" for a Simple Harmonic Oscillator),
        or a `celerite.terms.Term` can be passed in as a custom kernel
    cadence_mask : array-like
        A mask that will be applied to the cadences prior to constructing the GP
        model. For example, you can pass a boolean array of length `n_cadences`
        where `True` means that the cadence will be included in the noise model.
        You may also pass an array of indices. This option enables signals of
        interest (e.g. planet transits) to be excluded from the noise model,
        which will prevent over-fitting. By default, no cadences will be masked.
    sigma : int
        Number of sigma above which to remove outliers from the light curve
        when building the model
    """
    def __init__(self, lc, kernel="matern32", cadence_mask=None, sigma=5):
        log.debug('Initializing')
        if np.any([~np.isfinite(lc.flux), ~np.isfinite(lc.flux_err)]):
            log.warning("NaNs have been removed from the light curve.")
            self.lc = lc.remove_nans()
        else:
            self.lc = lc

        # Build cadence mask
        if cadence_mask is None:
            self.cadence_mask = np.ones(len(self.lc.time), dtype=bool)
        else:
            self.cadence_mask = cadence_mask

        # Add outliers to cadence mask
        self._bad_cadences = np.copy(~self.cadence_mask)
        self._bad_cadences |= self.lc.flatten().remove_outliers(sigma=sigma, return_mask=True)[1]
        self.diag = np.copy(self.lc.flux_err)
        self.diag[self._bad_cadences] *= 1e10  # This is faster than masking out flux values during the predict step

        # Store the unmasked light curve to fit later
        self._unmasked_lc = self.lc.copy()
        self._unmasked_diag = np.copy(self.diag)

        # Mask outlier cadences
        self.lc = self.lc[~self._bad_cadences]
        self.diag = self.diag[~self._bad_cadences]

        # Build the kernel
        log.debug("Building kernels")
        if isinstance(kernel, celerite.terms.Term):
            self.kernel = kernel
        else:
            kernel_str = validate_method(kernel, ["matern32", "sho"])
            self.kernel = self._build_kernel(kernel_str)

        # Compute the GP
        log.debug("Built kernels")
        self.gp = celerite.GP(self.kernel, mean=np.nanmean(self.lc.flux))#, fit_mean=True)
        self.gp.compute(self.lc.time, self.diag)
        self.initial_kernel = self.kernel
        self.optimized = False

    def _build_matern32_kernel(self, matern_bounds=None, jitter_bounds=None):
        """Helper function to construct the Matérn-3/2 kernel and set its starting
        values and bounds.

        Parameters
        ----------
        matern_bounds : dict
            Bounds for the Matern-3/2 kernel parameters (`log_sigma`, and
            `log_rho`)
        jitter_bounds : dict
            Bounds for the parameters of the jitter component of the kernel
            (`log_sigma`)

        Returns
        -------
        kernel : celerite.terms.Term
            Celerite kernel object (`celerite.terms.Term`)
        """
        log.debug('Building Matern3/2 Kernel')
        log_sigma = np.log(np.nanstd(self.lc.flux))
        log_rho = np.log(50.)
        log_sigma2 = np.log(np.nanmedian(self.lc.flux_err))
        log.debug('Created starting guesses')

        if matern_bounds is None:
            matern_bounds = {'log_sigma': (-2 + log_sigma, 2 + log_sigma),
                             'log_rho': (np.log(20.), np.log(75.))}
        if jitter_bounds is None:
            jitter_bounds = {'log_sigma':(-2 + log_sigma2, 2 + log_sigma2)}

        self.init_matern_bounds = matern_bounds
        self.init_jitter_bounds = jitter_bounds
        self.init_vals = {'log_sigma': log_sigma, 'log_rho': log_rho,
                          'log_sigma2': log_sigma2}

        kernel = celerite.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho, bounds=matern_bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=log_sigma2, bounds=jitter_bounds)
        return kernel

    def _build_sho_kernel(self, sho_bounds=None, jitter_bounds=None):
        """Helper function to construct the Simple Harmonic Oscillator kernel
        and set its starting values and bounds.

        Parameters
        ----------
        sho_bounds : dict
            Bounds for the Simple Harmonic Oscillator kernel parameters
            (`log_omega0`, `logS0`, and `logQ`)
        jitter_bounds : dict
            Bounds for the parameters of the jitter component of the kernel
            (`log_sigma`)

        Returns
        -------
        kernel : celerite.terms.Term
            Celerite kernel object (`celerite.terms.Term`)
        """
        log.debug('Building SHO Kernel')
        log_omega0 = np.log(2*np.pi / self.lc.normalize().to_periodogram(minimum_period=0.5, maximum_period=50).period_at_max_power.value)
        log_S0 = np.log(np.nanstd(self.lc.flux)**2)
        log_Q = np.log(10)
        log_sigma = np.log(np.nanmedian(self.lc.flux_err))
        log.debug('Created starting guesses')

        if sho_bounds is None:
            sho_bounds = {'log_S0': (-2 + log_S0, 2 + log_S0),
                          'log_Q': (0.2, 7),
                          'log_omega0': (np.log(2*np.pi/150), np.log(2*np.pi/0.1))}
        if jitter_bounds is None:
            jitter_bounds = {'log_sigma':(-2 + log_sigma, 2 + log_sigma)}

        self.init_sho_bounds = sho_bounds
        self.init_jitter_bounds = jitter_bounds
        self.init_vals = {'log_omega0': log_omega0, 'log_S0': log_S0,
                          'log_Q': log_Q, 'log_sigma': log_sigma}

        kernel = celerite.terms.SHOTerm(log_omega0=log_omega0, log_S0=log_S0,
                                        log_Q=log_Q, bounds=sho_bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=log_sigma, bounds=jitter_bounds)
        return kernel

    def _build_kernel(self, kernel_str="matern32"):
        """Returns a `celerite.terms.Term` object with reasonable initialization
        values.

        Parameters
        ----------
        kernel_str : str
            "matern32" for a Matérn-3/2 or "sho" for a Simple Harmonic Oscillator

        Returns
        -------
        kernel : celerite.terms.Term
            Celerite kernel object (`celerite.terms.Term`)
        """
        self.kernel_str = kernel_str
        if kernel_str == "matern32":
            kernel = self._build_matern32_kernel()
        elif kernel_str == "sho":
            kernel = self._build_sho_kernel()
        return kernel

    def _grad_neg_log_like(self, params, y):
        """Loss function and its gradient for likelihood of gp given a light curve."""
        self.gp.set_parameter_vector(params)
        ll, gll = self.gp.grad_log_likelihood(y)
        # Enforce bounds
        bounds = self._fetch_bounds_dict()
        for i, param in enumerate(params):
            bound_vals = list(bounds.values())[i]
            if param < bound_vals[0] or param > bound_vals[1]:
                return 1e25
        return -ll, -gll

    def optimize(self, method="L-BFGS-B"):
        """Function to optimize GP hyperparameters.

        Parameters
        ----------
        method : str
            Optimization method passed into `scipy.optimize.minimize`, default of
            "L-BFGS-B"

        Returns
        -------
        solution : scipy.optimize.optimize.OptimizeResult
            Output of optimizer
        """
        log.debug('Optimizing')
        solution = minimize(self._grad_neg_log_like, self.gp.get_parameter_vector(),
                            method=method, bounds=self.gp.get_parameter_bounds(),
                            jac=True, args=(self.lc.flux))
        self.optimized = True
        log.debug('Optimized')
        self.gp.set_parameter_vector(solution.x)

        self.solution = solution
        return solution

    def get_diagnostic_lightcurves(self, propagate_errors=False):
        """Returns dictonary of light curves.

        Parameters
        ----------
        propagate_errors : bool
            Boolean to optionally predict variance of the GP to probabilisticly
            estimate errors.

        Returns
        -------
        lightcurves : dict
            Dictionary of `~lightkurve.LightCurve` objects including
            `corrected_lc` and `gp_lc`
        """
        log.debug('Predicting...')
        self.gp.compute(self._unmasked_lc.time, self._unmasked_diag)
        if propagate_errors:
            log.debug('Propagating errors')
            gp_flux, gp_flux_var = self.gp.predict(self._unmasked_lc.flux, self._unmasked_lc.time)
            log.debug('Predicted.')

            corrected_lc = self._unmasked_lc.copy()
            corrected_lc.flux -= (gp_flux - np.mean(gp_flux))
            corrected_lc.flux_err = np.hypot(corrected_lc.flux_err, np.std(gp_flux_var))

            gp_lc = self._unmasked_lc.copy()
            gp_lc.flux = gp_flux
            gp_lc.flux_err = np.std(gp_flux_var)
        else:
            log.debug('Not propagating errors')
            gp_flux = self.gp.predict(self._unmasked_lc.flux, self._unmasked_lc.time, return_var=False, return_cov=False)
            log.debug('Predicted.')
            corrected_lc = self._unmasked_lc.copy()
            corrected_lc.flux -= (gp_flux - np.mean(gp_flux))

            gp_lc = self._unmasked_lc.copy()
            gp_lc.flux_err = 0
            gp_lc.flux = gp_flux

        return {'corrected': corrected_lc,
                'gp': gp_lc}

    def correct(self, propagate_errors=False):
        """Returns a `lightkurve.LightCurve` object with GP model for long-term
        trend removed.

        Parameters
        ----------
        propagate_errors : bool
            Boolean to optionally predict variance of the GP to probabilisticly
            estimate errors.

        Returns
        -------
        corrected_lc : lightkurve.LightCurve object
            A `~lightkurve.LightCurve` object with the long-term trend subtracted
            from the flux array
        """
        self.optimize()
        self.diagnostic_lightcurves = self.get_diagnostic_lightcurves(propagate_errors=propagate_errors)
        return self.diagnostic_lightcurves['corrected']

    def diagnose(self, ax=None, propagate_errors=False, **kwargs):
        """Show a diagnostic plot of the GP. Returns a matplotlib.pyplot.axes object.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object, default None
            Plot window to use
        propagate_errors : bool
            Boolean to optionally predict variance of the GP to probabilisticly
            estimate errors.
        kwargs: dict
            Keywords to pass to matplotlib.pyplot

        Returns
        -------
        ax : matplotlib.pyplot.axes object
        """
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            self._unmasked_lc.errorbar(ax=ax, zorder=1, label='Data', normalize=False)
            self._unmasked_lc[self._bad_cadences].scatter(ax=ax, zorder=2, color='r', marker='x', s=20, label='Rejected Outliers', normalize=False)

            if self.optimized:
                color = 'green'
                ax.set_title('Optimized')
            else:
                color='blue'
                ax.set_title('Pre-Optimization')

            gp = self.diagnostic_lightcurves['gp']
            if propagate_errors:
                ax.fill_between(gp.time, gp.flux - gp.flux_err, gp.flux + gp.flux_err, color=color)
            k = self.initial_kernel.get_parameter_dict()
            label = '\n'.join(['{}: {}'.format(key.split(':')[-1], np.round(k[key], 3))
                                        for key in k.keys()])
            gp.plot(ax=ax, color=color, label=label, normalize=False)
            ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper center', fancybox=True)
            plt.tight_layout()
        return ax

    def plot_distributions(self, ax=None, **kwargs):
        """Show a diagnostic plot to visualize the effect of optimization on the
        value of GP hyperparameters. Only works for default kernel options
        ("mater32" or "sho").

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object, default None
            Plot window to use
        kwargs: dict
            Keywords to pass to matplotlib.pyplot

        Returns
        -------
        ax : matplotlib.pyplot.axes object
        """
        start = self.init_vals
        bounds = self._fetch_bounds_dict()
        if self.kernel_str == 'matern32':
            finish = {'log_sigma':self.solution.x[0], 'log_rho':self.solution.x[1], 'log_sigma2':self.solution.x[2]}
        elif self.kernel_str == 'sho':
            finish = {'log_omega0':self.solution.x[0], 'log_S0':self.solution.x[1], 'log_Q':self.solution.x[2], 'log_sigma': self.solution.x[3]}

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots(1, len(bounds), figsize=(15,3))
            for i, param in enumerate(bounds):
                ax[i].set_title(param)
                ax[i].set_yticks([])
                ax[i].axvline(bounds[str(param)][0], c='b', label='Bounds', lw=3)
                ax[i].axvline(bounds[str(param)][1], c='b', lw=3)
                ax[i].axvline(start[str(param)], c='k', label='Initial Guess', lw=2)
                ax[i].axvline(finish[str(param)], c='r', label='Solution', lw=2)

        return ax

    def _fetch_bounds_dict(self):
        """ """
        if self.kernel_str == 'matern32':
            jitter = {}
            jitter['log_sigma2'] = self.init_jitter_bounds['log_sigma']
            bounds = {**self.init_matern_bounds, **jitter}
        elif self.kernel_str == 'sho':
            bounds = {**self.init_sho_bounds, **self.init_jitter_bounds}

        return bounds
