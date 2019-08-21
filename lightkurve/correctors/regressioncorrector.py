"""Defines RegressionCorrector"""
from __future__ import division, print_function

from tqdm import tqdm

import numpy as np
from scipy.optimize import minimize
from patsy import dmatrix

from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle

from .corrector import Corrector
from ..utils import LightkurveError, validate_method
from .. import LightCurve

class RegressionCorrector(Corrector):
    """Corrector class for regressing against input vectors.

    Parameters
    ----------
    lightcurve : `~lightkurve.lightcurve.LightCurve`
        The light curve object on which the SFF algorithm will be applied.
    """
    def __init__(self, lc, centroid_col=None, centroid_row=None, breakindex=None, period=None):
        self.lc = lc.normalize()
        self.raw_lc = self.lc
        self.flux = np.copy(self.lc.flux)
        self.flux_err = np.copy(self.lc.flux_err)
        self.time = np.copy(self.lc.time)
        self.model = np.ones(len(self.flux))
        self.period = period
        # Campaign break point in index
        if breakindex is None:
            self.breakindex = -1
        else:
            self.breakindex = breakindex


        for var in ['centroid_col', 'centroid_row']:
            if locals()[var] is None:
                if not hasattr(lc, var):
                    raise LightkurveError('Input light curve does not have a {0} attribute. '
                                          'Please pass a {0}'.format(var))
                elif (~np.isfinite(getattr(lc, var))).all():
                    raise LightkurveError('Input light curve {0} attribute is all nans. '
                                          'Please pass a valid {0}'.format(var))
                else:
                    setattr(self, var, getattr(lc, var))
            else:
                setattr(self, var, locals()[var])



    def _normalize_and_split_design_matrix(self, design_matrix, breakindex):
        """ Split a design matrix at a breakindex, and normalize all vectors.

        Parameters
        ----------
        design_matrix : np.ndarray
            Design matrix, with dimensions time x nvectors
        breakindex : int
            Index of a point to "break" the data at. Design matrix will be
            split into two halves, with zeros before or after the breakindex.

        Returns
        -------
        normalized_design_matrix : np.ndarray
            Normalized and split design matrix
        """
        components = np.copy(design_matrix)
        components -= np.atleast_2d(np.nanmean(components, axis=0))
        components /= np.atleast_2d(np.nanstd(components, axis=0))
        components *= self.lc.flux.std()

    #        components /= len(components) * 2


        stack = []
        for idx in np.array_split(np.arange(len(components)), [breakindex]):
            mask = np.in1d(np.arange(len(components)), idx)
            if mask.sum() == 0:
                continue
            window = np.copy(components)
            window[~mask] *= np.nan
            stack.append(window)
            zeros = np.atleast_2d(np.zeros(len(components))).T
            zeros[~mask] *= np.nan
            stack.append(zeros)
        #mean = np.atleast_2d(np.ones(len(components))).T * mean
        #stack.append(mean)
        components = np.hstack(stack)
        components /= len(components.T)
        components += 1

        return np.nan_to_num(components)

    def _solve_weights(self, X, cadence_mask=None):
        """Compute the weights of a given input matrix using np.linalg.solve

        Parameters:
        -----------
        X : np.ndarray
            Input matrix
        flux : np.ndarray
            Flux to fit design matrix to
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns:
        --------
        w : np.ndarray
            Best fit weights of each vector.
        model : np.ndarray
            The best fit model to the data (X dot w)
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(flux), bool)
        A = np.dot(X[cadence_mask].T, X[cadence_mask])
        B = np.dot(X[cadence_mask].T, (self.flux[cadence_mask] - np.median(self.flux[cadence_mask])))
        w = np.linalg.solve(A, B)
        model = np.dot(X, w)
        return w, model

    def _basic_design_matrix(self):
        """Build a basic design matrix based on the PSF position.

        Builds a design matrix of:

            arclength, arclength**2, arclength**3, darclength/dt
            Column**4, Column**3, Column**2, Column
            Row**4, Row**3, Row**2, Row
            Column**4 Row**3, Column**4 Row **2, Column**4 Row, Column**3 Row**2, Column**2 Row, Column Row
            Row**4 Column**3, Row**4 Column**2, Row**4 Column, Row**3 Column**2, Row**3 Column, Row**2 Column
            Vector Of Ones
        """
        row, col = self.centroid_col - self.centroid_col.min(), self.centroid_row - self.centroid_row.min()
        build_components = lambda X, Y: np.array([
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5,
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2),
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**1.5,
                                                 np.gradient(((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5),
                                                 X**4, X**3, X**2, X,
                                                 Y**4, Y**3, Y**2, Y,
                                                 X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                                                 Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X]).T

        return build_components(row, col)


    def _optimize_lomb_scargle(self, design_matrix, cadence_mask=None, period=None, nterms=3, bspline=False,
                                pmin=0.1, pmax=300, nperiod=30, n_knots=10):
        """ Find the best long term trend using sine curves, fitting for the optimum period.

        Will add sine curves of order `nterms` to the design matrix, and will fit
        the period of the sine curves automatically. Pass in a `period` to
        specify the period of the long term trend.

        The best period is found by minimizing the chi**2 fit, using scipy.optimize.minimize.
        Multiple start periods are passed in for this optimization, to avoid finding
        a local minimum.

        Parameters:
        -----------
        design_matrix : np.ndarray
            Design matrix, with dimensions time x nvectors
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        period : float (optional)
            Period to fit the sinusoids at. If None, the optimum period will be found
        nterms : int (default 3)
            Number of terms to fit the sinusoids. More terms are more flexible,
            but take longer to fit.
        bspline : bool (default False)
            Whether to also fit a bspline. See the _optimize_spline docstring for details
        pmin : float (default 0.1)
            Minimum period to try to fit
        pmax : float (default 300)
            Maximum period to try to fit
        nperiod : int (default 30)
            Number of times to attempt to find the minimum chi**2 period,
            at different starting values.
        n_knots : int
            Number of knots to use for the optional spline. These will be evenly spaced.

        Returns:
        --------
        w : np.ndarray
            Best fit weights of each vector.
        dm : np.ndarray
            The input design matrix contatinated with the lomb-scargle design matrix.
        model : np.ndarray
            The best fit model to the data (X dot w)
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(lc.flux), bool)
        ls = LombScargle(self.time, self.flux, nterms=nterms)
        if bspline:
            spline_dm = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=False) - 1".format(n_knots), {"x": self.time}))
            ls = np.hstack([ls, spline_dm])

        def func(params, cadence_mask, return_model=False):
            """Function to optimize period"""
            ls_dm = ls.design_matrix(1/params[0])[:, 1:]
            dm = np.hstack([design_matrix, ls_dm, np.atleast_2d(np.ones(len(self.flux))).T])
            w, model = self._solve_weights(dm, cadence_mask=cadence_mask)
            if return_model:
                return w, dm, model
            resids = self.flux - (model + 1)
            return np.sum(((resids)**2/self.flux_err**2))/len(resids)

        if period is None:
            # Optimize period using scipy.optimize.minimize
            x, fun = np.zeros(nperiod), np.zeros(nperiod)
            for idx, start in enumerate(tqdm(np.logspace(np.log10(pmin), np.log10(pmax), nperiod), desc='Finding Optimum LS Period')):
                res = minimize(func, [start], method='Powell', args=cadence_mask)
                x[idx], fun[idx] = res.x, res.fun
            best_period = x[x < pmax][np.argmin(fun[x < pmax])]
        else:
            res = minimize(func, [period], method='Powell', args=cadence_mask)
            best_period = res.x
        w, dm, model = func([best_period], cadence_mask=cadence_mask, return_model=True)
        self.period = best_period
        return w, dm, model

    def _optimize_spline(self, design_matrix, cadence_mask=None, n_knots=10):
        """Find the best fitting bspline to the long term trends in the light curve.

        The optimization here is done with a simple linear regression using np.linalg.solve.

        Parameters:
        -----------
        design_matrix : np.ndarray
            Design matrix, with dimensions time x nvectors
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        n_knots : int
            Number of knots to use for the spline. These will be evenly spaced.
        Returns:
        --------
        w : np.ndarray
            Best fit weights of each vector.
        dm : np.ndarray
            The input design matrix contatinated with the lomb-scargle design matrix.
        model : np.ndarray
            The best fit model to the data (X dot w)
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(lc.flux), bool)
        spline_dm = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=False) - 1".format(n_knots), {"x": self.time}))
        dm = np.hstack([design_matrix, spline_dm, np.atleast_2d(np.ones(len(self.flux))).T])
        w, model = self._solve_weights(dm, cadence_mask=cadence_mask)
        return w, dm, model

    def correct(self, design_matrix=None, cadence_mask=None, method='spline', preserve_trend=True, sigma=5, niters=5, timescale=3, normalize_and_split=True, **kwargs):
        """Find the best fit correction for the light curve.

        Parameters:
        -----------
        design_matrix : np.ndarray
            Design matrix, with dimensions time x nvectors
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        method : str (default 'spline')
            Method to remove long term trends. Options: spline, lombscargle
        preserve_trend: bool (default True)
            Whether to preserve or remove long term trends from the returned
            lightcurve
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        timescale : float (default 3)
            Timescale on which to remove long term trends when using the spline corrector.
            Timescale establishes the knotspacing for the spline.
        normalize_and_split: bool (default True)
            Whether to normalize and split the input design matrix. If True, the
            input design matrix will be normalized and split at self.breakindex.
        **kwargs: dict
            Keyword arguments to pass to the optimizer for each `method`.

        Returns:
        -------
        corrected : lightkurve.LightCurve
            Corrected light curve, with noise removed.
        """

        self.method = validate_method(method, ['spline', 'lombscargle'])

        if design_matrix is None:
            design_matrix = self._normalize_and_split_design_matrix(self._basic_design_matrix(), self.breakindex, **kwargs)
        elif normalize_and_split:
            if design_matrix.shape[0] != len(self.flux):
                raise LightkurveError('Design matrix must have shape ncadences x nvectors. ({} x n)'.format(len(self.flux)))
            design_matrix = self._normalize_and_split_design_matrix(np.copy(design_matrix), self.breakindex, **kwargs)
        dm = np.copy(design_matrix)
        if cadence_mask is None:
            mask = np.ones(len(self.time), bool)
        else:
            mask = np.copy(cadence_mask)

        n_knots = int((self.time[-1] - self.time[0])/timescale)
        n_knots = np.max([n_knots, 3])
        for count in range(niters):
            if self.method == 'spline':
                w, dm2, model = self._optimize_spline(dm, cadence_mask=mask, n_knots=n_knots)
            if self.method == 'lombscargle':
                w, dm2, model = self._optimize_lomb_scargle(dm, cadence_mask=mask, n_knots=n_knots, period=self.period)
            if count != niters - 1:
                mask &= ~sigma_clip(self.flux - model, sigma=sigma).mask

        noise = LightCurve(self.time, np.dot(w[:len(dm.T)], dm.T))
        noise.flux -= np.median(noise.flux)
        long_term = LightCurve(self.time, np.dot(w[len(dm.T):], dm2[:, len(dm.T):].T))
        long_term.flux -= np.median(long_term.flux)
        if preserve_trend:
            corrected = self.lc - noise.flux
        else:
            corrected = self.lc - model
        self.diagnostic_lightcurves = {'noise':noise, 'long_term':long_term, 'corrected':corrected}
        self.cadence_mask = mask
        self.design_matrix = dm2
        return corrected


    def diagnose(self):
        """ Produce diagnostic plots to assess the effectiveness of the correction. """
        ax = self.lc.plot(normalize=False, label='Original', alpha=0.4)
        (self.diagnostic_lightcurves['noise'] + 1).plot(ax=ax, c='b', lw=0.4, label='Noise Model')
        if self.method == 'spline':
            label = 'spline'
        if self.method == 'lombscargle':
            label = 'LS Period : {:5.2f}'.format(self.period)
        (self.diagnostic_lightcurves['long_term'] + 1).plot(ax=ax, c='r', lw=1, label='Long Term Trend ({})'.format(label))

        ax = self.lc.plot(normalize=False, alpha=0.2, label='Original')
        self.lc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x', s=10, label='Outliers', ax=ax)

        self.diagnostic_lightcurves['corrected'].plot(normalize=True, label='Corrected', ax=ax, c='k')
        return
