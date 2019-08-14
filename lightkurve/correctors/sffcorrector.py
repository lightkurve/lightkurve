"""Defines SFFCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from scipy import linalg, interpolate
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

from .corrector import Corrector
from .gpcorrector import GPCorrector

log = logging.getLogger(__name__)

__all__ = ['SFFCorrector']

import celerite
from .. import MPLSTYLE
from ..utils import LightkurveError

class SFFCorrector(Corrector):

    def __init__(self, lc, centroid_col=None, centroid_row=None, breakpoint=None):
        self.lc = lc.remove_nans()
        self.optimized = False
        self.flux = np.copy(self.lc.flux)
        self.flux_err = np.copy(self.lc.flux_err)
        self.time = np.copy(self.lc.time)
        self.model = np.ones(len(self.flux))
        self.type = type

        # Campaign break point in cadence number
        if breakpoint is None:
            self.breakpoint = self._get_break_point(self.lc.campaign)
        else:
            self.breakpoint = breakpoint

        # Campaign break point in index
        self.breakindex = np.argmin(np.abs(self.lc.cadenceno - self.breakpoint))
        # Input validation
        if centroid_col is None:
            try:
                self.centroid_col = self.lc.centroid_col
            except AttributeError:
                raise ValueError('`centroid_col` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')
        else:
            self.centroid_col = centroid_col

        if centroid_row is None:
            try:
                self.centroid_row = self.lc.centroid_row
            except AttributeError:
                raise ValueError('`centroid_row` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')
        else:
            self.centroid_row = centroid_row
        c, r = self.centroid_col - np.min(self.centroid_col) + 1, self.centroid_row - np.min(self.centroid_row) + 1
        self.arc = (c**2 + r**2)**0.5
#        self.arc += 1

    def create_design_matrix(self, window_points=None):
        if window_points is None:
            if hasattr(self, 'window_points'):
                window_points = self.window_points
            elif hasattr(self, 'breakindex'):
                window_points = [self.breakindex]
            else:
                raise ValueError('Please pass window points.')


        if not hasattr(window_points, '__iter__'):
            window_points = [window_points]


        build_components = lambda X, Y, T: np.array([
                                             ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5,
                                             ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2),
                                             ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**1.5,
                                             X**4, X**3, X**2, X,
                                             Y**4, Y**3, Y**2, Y,
                                             X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                                             Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X, np.ones(len(T))]).T

        # whiten
        c, r, t = np.copy(self.centroid_col), np.copy(self.centroid_row), np.copy(self.time)
        c, r, t = c - c.mean(), r - r.mean(), t - t.mean()
        c, r, t = c / c.std(), r / r.std(), t / t.std()

        components = build_components(c, r, t)
        stack = []
        for idx in np.array_split(np.arange(len(self.time)), [self.breakindex]):
            mask = np.in1d(np.arange(len(self.time)), idx)
            if mask.sum() == 0:
                continue
            window = np.copy(components)
            window[~mask] *= np.nan
            stack.append(window)


        arcstack = []
        for idx in np.array_split(np.arange(len(self.time)), window_points):
            mask = np.in1d(np.arange(len(self.time)), idx)
            if mask.sum() == 0:
                continue
            arc_masked = np.copy(self.arc)
            arc_masked[~mask] *= np.nan
            ones = np.ones(len(self.time))
            ones[~mask] *= np.nan
            window = np.asarray([ones, arc_masked, arc_masked**2, arc_masked**3, arc_masked**4]).T
            arcstack.append(window)

        stack.append(np.hstack(arcstack))
        components = np.hstack(stack)
        components -= np.atleast_2d(np.nanmean(components, axis=0))
        std = np.nanstd(components, axis=0)
        std[std == 0] = 1
        components /= np.atleast_2d(std)
        components *= self.lc.estimate_cdpp() * 1e-6
        # components /= np.nanmedian(np.nansum(components == 0, axis=1))
        components += 1
        components *= self.lc.flux.mean()
        return np.nan_to_num(components)

    def _solve_weights(self, design_matrix, gp_corrector, l2_term):
        """Function to perform the analytic computation of the PLD algorithm.
        Returns a noise model light curve.

        Parameters
        ----------
        design_matrix : 2D numpy array
            Matrix containing suitable regressors for the systematics noise model
            with shape (n_cadences, n_pca_terms*pld_order)
        gp_corrector : lightkurve.GPCorrector object or None
            Lightkurve GPCorrector object used to estimate long-term astrophysical
            trend in the observation
        l2_term : float
            It's complicated

        Returns
        -------
        noise_model : numpy array
            1D numpy array for the noise model light curve
        """
        gp = gp_corrector.gp
        X = design_matrix
        A = np.dot(X.T, gp.apply_inverse(X))
        # To ensure the weights can be solved for, we need to perform L2 regularization
        # to avoid an ill-conditioned matrix A. Here we add small values along the
        # diagonal of matrix A to reduce its condition number and  improve its
        # numerical stability
        A[np.diag_indices_from(A)] += l2_term
        B = np.dot(X.T, gp.apply_inverse(self.lc.flux[:, None])[:, 0])
        # Solve for the weights and compute the final model
        w = np.linalg.solve(A, B)
        noise_model = np.dot(X, w)

        return noise_model


    def _grad_neg_log_like(self, params, design_matrix, gp_corrector, l2_term):
        """Gradient of loss function to improve model optimization."""
        gp_corrector.gp.set_parameter_vector(params)
        noise_model = self._solve_weights(design_matrix, gp_corrector, l2_term=l2_term)
        ll, gll = gp_corrector.gp.grad_log_likelihood(self.lc.flux - noise_model)
        return -ll, -gll

    def optimize(self, design_matrix, gp_corrector, l2_term, method="L-BFGS-B"):
        """Function to optimize GP hyperparameters simultaneously with fitting
        the PLD noise model.

        Parameters
        ----------
        design_matrix : 2D numpy array or None
            Matrix containing suitable regressors for the systematics noise model
            with shape (n_cadences, n_pca_terms*pld_order). If set to None, a
            design matrix will be generated with default values
        gp_corrector : lightkurve.GPCorrector object or None
            Lightkurve GPCorrector object used to estimate long-term astrophysical
            trend in the observation
        method : str
            Optimization method passed into `scipy.optimize.minimize`, default of
            "L-BFGS-B"

        Returns
        -------
        gp_corrector : lightkurve.GPCorrector object or None
            Lightkurve GPCorrector object used to estimate long-term astrophysical
            trend in the observation with optimized hyperparameters
        """
        self.optimized = True

        # find a maximum-likelihood solution
        solution = minimize(self._grad_neg_log_like, gp_corrector.gp.get_parameter_vector(),
                            method=method, bounds=gp_corrector.gp.get_parameter_bounds(),
                            jac=True, args=(design_matrix, gp_corrector, l2_term))
        # set the GP parameters to the optimization output
        gp_corrector.gp.set_parameter_vector(solution.x)
        self.gp_corrector = gp_corrector
        self._gp_solution = solution
        return self.gp_corrector

    def _get_window_points(self, windows):
        ''' Build window points, based on where thrusters are fired. '''

        def _get_thruster_firings(arc):
            ''' Find locations where K2 fired thrusters
            Parameters:
            ----------
            arc : np.ndarray
                arclength as a function of time
            Returns:
            -------
            thrusters: np.ndarray of bools
                True at times where thrusters were fired.
            '''
            # Rate of change of rate of change of arclength wrt time
            d2adt2 = (np.gradient(np.gradient(arc)))
            # Fit a nice Gaussian, most points lie in a tight region, thruster firings are outliers
            g = models.Gaussian1D(amplitude=100, mean=0, stddev=0.01)
            fitter = fitting.LevMarLSQFitter()
            h = np.histogram(d2adt2, np.arange(-0.5, 0.5, 0.0001), density=True);
            xbins = h[1][1:] - np.median(np.diff(h[1]))
            g = fitter(g, xbins, h[0], weights=h[0]**0.5)

            def _start_and_end(type):
                ''' Find points at the start or end of a roll
                '''
                if type == 'start':
                    thrusters = d2adt2 < (g.stddev * -5)
                if type == 'end':
                    thrusters = d2adt2 > (g.stddev * 5)
                # Pick the best thruster in each cluster
                idx = np.array_split(np.arange(len(thrusters)), np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0])
                m = np.array_split(thrusters, np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0])
                th = []
                for jdx in range(len(idx)):
                    if m[jdx].sum() == 0:
                        th.append(m[jdx])
                    else:
                        th.append((np.abs(np.gradient(arc)[idx[jdx]]) == np.abs(np.gradient(arc)[idx[jdx]][m[jdx]]).max()) & m[jdx])
                thrusters = np.hstack(th)
                return thrusters

            # Get the start and end points
            thrusters = np.asarray([_start_and_end('start'), _start_and_end('end')])
            thrusters = thrusters.any(axis=0)

            # Take just the first point.
            thrusters = (np.gradient(np.asarray(thrusters, int)) >= 0) & thrusters
            return thrusters

        thrusters = _get_thruster_firings(self.arc)
        thrusters[self.breakindex] = True
        thrusters = np.where(thrusters)[0]
        window_points = np.append(np.linspace(0, self.breakindex + 1, windows//2 + 1, dtype=int)[1:],
                              np.linspace(self.breakindex + 1, len(self.arc), windows//2 + 1, dtype=int)[1:-1])
        window_points[np.argmin((window_points - self.breakindex + 1)**2)] = self.breakindex + 1

        """if self.breakindex != 0:
            window_points = np.append(np.linspace(0, self.breakindex + 1, windows//2 + 1, dtype=int)[1:],
                                  np.linspace(self.breakindex + 1, len(self.arc), windows//2 + 1, dtype=int)[1:-1])
            window_points[np.argmin((window_points - self.breakindex + 1)**2)] = self.breakindex + 1
        else:
            window_points = np.linspace(0, len(self.flux) + 1, windows)
        window_points = [thrusters[np.argmin(np.abs(wp - thrusters))] + 1 for wp in window_points]
        # window_points[0] = 0"""
        return window_points

    def correct(self, cadence_mask=None, preserve_trend=True, design_matrix=None,
                gp_corrector=None, l2_term=None, windows=20, bins=10, window_points=None, **kwargs):
        """Returns a `lightkurve.LightCurve` object with model for motion noise
        removed.

        Parameters
        ----------
        cadence_mask : array-like
            A mask that will be applied to the cadences prior to constructing
            the detrending model. For example, you can pass a boolean array
            of length `n_cadences` where `True` means that the cadence will be
            included in the noise model. You may also pass an array of indices.
            This option enables signals of interest (e.g. planet transits)
            to be excluded from the noise model, which will prevent over-fitting.
            By default, no cadences will be masked.
        preserve_trend : bool
            Option to remove long-term trend fit by GP. By default, the
            astrophysical signal is preserved, but can be subtracted out to
            robustly flatten the output light curve.
        design_matrix : 2D numpy array or None
            Matrix containing suitable regressors for the systematics noise model
            with shape (n_cadences, n_pca_terms*pld_order). If set to None, a
            design matrix will be generated
        gp_corrector : lightkurve.GPCorrector object or None
            Lightkurve GPCorrector object used to estimate long-term astrophysical
            trend in the observation. If set to None, a GP will be generated
        **kwargs : dict
            Keyword arguments for the `~lightkurve.GPCorrector` object

        Returns
        -------
        corrected_lc : lightkurve.LightCurve object
            A `~lightkurve.LightCurve` object with the noise model subtracted
            from the flux array
        """
        # Create final optimized model
        if gp_corrector is None:
            gp_corrector = GPCorrector(self.lc, cadence_mask=cadence_mask, sigma=1e10, **kwargs)
        elif isinstance(gp_corrector, celerite.GP):
            gp_corrector = GPCorrector(self.lc, cadence_mask=cadence_mask, sigma=1e10, kernel=gp_corrector.kernel, **kwargs)

        self.windows = windows
        self.bins = bins

        if window_points is None:
            if self.windows <= 1:
                self.window_points = [self.breakindex + 1]
            else:
                self.window_points = self._get_window_points(windows)
        else:
            self.window_points = window_points


        if design_matrix is None:
            design_matrix = self.create_design_matrix(self.window_points)


        # The L2 regularization term should roughly be equal to the inverse of
        # the amplitude of the signals we want the noise model to fit
        if l2_term is None:
            l2_term = 1 / (np.nanmedian(self.lc.flux) * self.lc.estimate_cdpp() * 1e-6)**2
            log.debug("Setting l2_term to {}".format(l2_term))

        # Optimize the GP
#        if not self.optimized:
        niters = 3
        for count in range(niters):
            gp_corrector = self.optimize(design_matrix, gp_corrector, l2_term=l2_term)


            # Create noise model LightCurve
            noise_lc = self.lc.copy()
            noise_lc.flux = self._solve_weights(design_matrix, gp_corrector, l2_term)
            # Create corrected LightCurve
            corrected_lc = self.lc.copy()
            corrected_lc.flux -= noise_lc.flux
            corrected_lc.flux += np.nanmean(noise_lc.flux)
            # Create GP LightCurve
            gp_lc = self.lc.copy()
            if count <= niters - 1:
                mu = gp_corrector.gp.predict(corrected_lc.flux, corrected_lc.time,
                                             return_cov=False, return_var=False)
                gp_lc.flux = mu
                outliers = (corrected_lc - (gp_lc.flux - np.nanmean(gp_lc.flux))).remove_outliers(sigma=5, return_mask=True)[1]
                gp_corrector.diag[outliers] *= 1e10
            else:
                mu, var = gp_corrector.gp.predict(corrected_lc.flux, corrected_lc.time,
                                             return_cov=True, return_var=False)
                gp_lc.flux = mu
                gp_lc.flux_err = var**0.5
                corrected_lc.flux_err = (corrected_lc.flux_err**2 + gp_lc.flux_err**2)**0.5
        # Optionally remove long term trend fit by GP
        if not preserve_trend:
            corrected_lc.flux -= (gp_lc.flux - np.nanmean(gp_lc.flux))

        self.diagnostic_lightcurves = {'noise': noise_lc,
                                       'corrected': corrected_lc,
                                       'gp': gp_lc}
        return self.diagnostic_lightcurves['corrected']

    def diagnose(self, ax=None):
        """Diagnostic plotting function to assess performance of the PLD de-trending."""
        if not self.optimized:
            raise LightkurveError("You need to call the `optimize` or `correct` method before diagnosing.")

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots(3, figsize=(8.485, 12))

        # raw and corrected
        self.lc.scatter(ax=ax[0], c='k', label='{} (Raw Light Curve)'.format(self.lc.label),
                            normalize=False, alpha=0.2, s=0.5)

        for idx in np.array_split(np.arange(len(self.time)), self.window_points):
            self.lc[idx].scatter(ax=ax[0], label='', normalize=False, alpha=0.4, s=0.5)

        self.diagnostic_lightcurves['corrected'].scatter(ax=ax[0], c='k',
                                                         label='{} (PLD-Corrected)'.format(self.lc.label),
                                                         normalize=False, s=0.5)

        # raw and gp
        self.lc.scatter(ax=ax[1], c='r', label='{} (Raw Light Curve)'.format(self.lc.label),
                            normalize=False, alpha=0.5)
        self.diagnostic_lightcurves['gp'].plot(ax=ax[1], c='k', lw=1,
                                               label='{} (GP Trend)'.format(self.lc.label),
                                               normalize=False)

        # raw and corrected
        self.diagnostic_lightcurves['noise'].scatter(ax=ax[2], c='k',
                                                     label='{} (Noise Model)'.format(self.lc.label),
                                                     normalize=False)
        return ax
