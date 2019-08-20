"""Defines SFFCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from patsy import dmatrix
from scipy import linalg, interpolate
from matplotlib import pyplot as plt

from astropy.stats import sigma_clip
from astropy.modeling import models, fitting

from .corrector import Corrector
from .regressioncorrector import RegressionCorrector

from ..utils import LightkurveError, validate_method
from .. import LightCurve



log = logging.getLogger(__name__)

__all__ = ['SFFCorrector']


class SFFCorrector(RegressionCorrector):
    """Implements the Self-Flat-Fielding (SFF) systematics removal method.

    This method is described in detail by Vanderburg and Johnson (2014).
    Briefly, the algorithm implemented in this class can be described
    as follows

       (1) Rotate the centroid measurements onto the subspace spanned by the
           eigenvectors of the centroid covariance matrix
       (2) Fit a polynomial to the rotated centroids
       (3) Compute the arclength of such polynomial
       (4) Fit a BSpline of the raw flux as a function of time
       (5) Normalize the raw flux by the fitted BSpline computed in step (4)
       (6) Bin and interpolate the normalized flux as a function of the arclength
       (7) Divide the raw flux by the piecewise linear interpolation done in step (6)
       (8) Set raw flux as the flux computed in step (7) and repeat
       (9) Multiply back the fitted BSpline

    Parameters
    ----------
    lightcurve : `~lightkurve.lightcurve.LightCurve`
        The light curve object on which the SFF algorithm will be applied.

    Examples
    --------
    """
    def __init__(self, *args, **kwargs):
        super(SFFCorrector, self).__init__(*args, **kwargs)
        self.arc = ((self.centroid_col - self.centroid_col.min())**2 + (self.centroid_row - self.centroid_row.min())**2)**0.5
        self.arc += 1

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

        if self.breakindex != 0:
            window_points = np.append(np.linspace(0, self.breakindex + 1, windows//2 + 1, dtype=int)[1:],
                                  np.linspace(self.breakindex + 1, len(self.arc), windows//2 + 1, dtype=int)[1:-1])
            window_points[np.argmin((window_points - self.breakindex + 1)**2)] = self.breakindex + 1
        else:
            window_points = np.linspace(0, len(self.flux) + 1, windows)
        window_points = [thrusters[np.argmin(np.abs(wp - thrusters))] + 1 for wp in window_points]
        if window_points[0] < 10:
            window_points[0] = 0
        return window_points


    def _get_window_design_matrix(self):
        stack = []
        for idx in np.array_split(np.arange(len(self.arc)), self.window_points):
            mask = np.in1d(np.arange(len(self.arc)), idx)
            if mask.sum() == 0:
                continue
            window = np.zeros((len(self.arc), self.bins)) * np.nan
            window[mask] = np.asarray(dmatrix("bs(x, df={}, degree=2, include_intercept=True) - 1".format(self.bins), {"x": self.arc[mask]}))
            stack.append(window)
        window_dm = np.hstack(stack)
        return np.nan_to_num(window_dm + 1)


    def _get_window_design_matrix(self):
        stack = []
        window = np.zeros((len(self.arc), self.bins)) * np.nan
        window = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=True) - 1".format(self.windows), {"x": self.arc}))
        stack.append(window)

        for idx in np.array_split(np.arange(len(self.arc)), self.window_points):
            mask = np.in1d(np.arange(len(self.arc)), idx)
            if mask.sum() == 0:
                continue
            window = np.zeros((len(self.arc), self.bins)) * np.nan
            window[mask] = np.asarray(dmatrix("bs(x, df={}, degree=2, include_intercept=True) - 1".format(self.bins), {"x": self.arc[mask]}))
            stack.append(window)
        window_dm = np.hstack(stack)

        window_dm = np.hstack(stack)
        return np.nan_to_num(window_dm + 1)

    def correct(self, design_matrix=None, cadence_mask=None, method='spline', preserve_trend=True, sigma=5, niters=5, timescale=3, bins=10, windows=20):
        self.method = validate_method(method, ['spline', 'lombscargle'])

        if design_matrix is None:
            design_matrix = self._normalize_and_split_design_matrix(self._basic_design_matrix(), self.breakindex)
        dm = np.copy(design_matrix)
        if cadence_mask is None:
            mask = np.ones(len(self.time), bool)
        else:
            mask = np.copy(cadence_mask)


        self.bins = np.max([bins, 2])
        self.windows = windows
        self.window_points = self._get_window_points(windows)

        window_dm = self._get_window_design_matrix()
        dm = np.hstack([dm, window_dm])
#        dm = window_dm

        n_knots = int((self.time[-1] - self.time[0])/timescale)
        n_knots = np.max([n_knots, 3])
        for iter in range(niters):
            if self.method == 'spline':
                w, dm2, model = self._optimize_spline(dm, cadence_mask=mask, n_knots=n_knots)
            if self.method == 'lombscargle':
                w, dm2, model = self._optimize_lomb_scargle(dm, cadence_mask=mask, n_knots=n_knots, start=self.period)
            if iter != niters - 1:
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
        self.weights = w
        # window_design_matrix = window_dm
        # window_weights = w[len(dm.T)-len(window_dm.T):len(dm.T)]
        # self.window_model = np.dot(window_weights, window_design_matrix.T)
        # self.window_model -= np.median(self.window_model)

        return corrected


    def diagnose(self, ax=None):
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
#         f = self.flux - self.diagnostic_lightcurves['long_term'].flux
#         n = self.diagnostic_lightcurves['noise'].flux
#         window_dm = self._get_window_design_matrix()
#         for idx in np.array_split(np.arange(len(self.arc)), self.window_points):
#             mask = np.in1d(np.arange(len(self.arc)), idx)
#             if mask.sum() == 0:
#                 continue
#             plt.figure()
#             plt.scatter(self.arc[mask], f[mask], s=0.5, c='k')
#             plt.scatter(self.arc[mask & ~self.cadence_mask], f[mask & ~self.cadence_mask], s=10, c='r', marker='x')
#
#             plt.errorbar(self.arc[mask], f[mask], self.flux_err[mask], c='k', ls='')
#
#             s = np.argsort(self.arc[mask])
#             plt.scatter(self.arc[mask][s], n[mask][s] + 1, c=self.time[mask], vmin=self.time[0], vmax=self.time[-1], cmap='coolwarm')
#
# #            break
        return
