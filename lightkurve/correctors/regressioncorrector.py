"""Defines RegressionCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from scipy import linalg, interpolate
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from astropy.stats import sigma_clip
from astropy.timeseries import LombScargle

from tqdm import tqdm

from itertools import combinations_with_replacement as multichoose


from .corrector import Corrector

from patsy import dmatrix
import celerite
from .. import MPLSTYLE
from ..utils import LightkurveError, validate_method
from .. import LightCurve

class RegressionCorrector(Corrector):

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
        components = np.copy(design_matrix)
        components -= np.atleast_2d(np.nanmean(components, axis=0))
        components /= np.atleast_2d(np.nanstd(components, axis=0))
        components *= self.lc.estimate_cdpp() * 1e-6

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

    def _solve_weights(self, X, flux, cadence_mask=None):
        if cadence_mask is None:
            cadence_mask = np.ones(len(flux), bool)
        A = np.dot(X[cadence_mask].T, X[cadence_mask])
        B = np.dot(X[cadence_mask].T, (flux[cadence_mask] - np.median(flux[cadence_mask])))
        w = np.linalg.solve(A, B)
        model = np.dot(X, w)
        return w, model

    def _basic_design_matrix(self):
        row, col = self.centroid_col - self.centroid_col.min(), self.centroid_row - self.centroid_row.min()
        build_components = lambda X, Y: np.array([
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5,
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2),
                                                 ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**1.5,
                                                 np.gradient(((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5),
                                                 np.gradient(((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)),
                                                 X**4, X**3, X**2, X,
                                                 Y**4, Y**3, Y**2, Y,
                                                 X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                                                 Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X]).T

        return build_components(row, col)


    def _optimize_lomb_scargle(self, design_matrix, cadence_mask=None, n_knots=20, start=None):
        if cadence_mask is None:
            cadence_mask = np.ones(len(lc.flux), bool)
        spline_dm = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=False) - 1".format(n_knots), {"x": self.time}))

        ls = LombScargle(self.time, self.flux, nterms=3)

        def func(params, cadence_mask, return_model=False):
            ls_dm = ls.design_matrix(1/params[0])[:, 1:]
            dm = np.hstack([design_matrix, spline_dm, ls_dm, np.atleast_2d(np.ones(len(self.flux))).T])
            w, model = self._solve_weights(dm, self.flux, cadence_mask=cadence_mask)
            if return_model:
                return w, dm, model
            resids = self.flux - (model + 1)
            return np.sum(((resids)**2/self.flux_err**2))/len(resids)

        if start is None:
            x, fun = np.zeros(30), np.zeros(30)
            for idx, start in enumerate(tqdm(np.logspace(0, 2, 30), desc='Finding Optimum LS Period')):
                res = minimize(func, [start], method='Powell', args=cadence_mask)
                x[idx], fun[idx] = res.x, res.fun
            best_period = x[x < 300][np.argmin(fun[x < 300])]
        else:
            res = minimize(func, [start], method='Powell', args=cadence_mask)
            best_period = res.x
        w, dm, model = func([best_period], cadence_mask=cadence_mask, return_model=True)
        self.period = best_period
        return w, dm, model

    def _optimize_spline(self, design_matrix, cadence_mask=None, n_knots=20):
        if cadence_mask is None:
            cadence_mask = np.ones(len(lc.flux), bool)
        spline_dm = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=False) - 1".format(n_knots), {"x": self.time}))
        dm = np.hstack([design_matrix, spline_dm, np.atleast_2d(np.ones(len(self.flux))).T])
        w, model = self._solve_weights(dm, self.flux, cadence_mask=cadence_mask)
        return w, dm, model

    def correct(self, design_matrix=None, cadence_mask=None, method='spline', preserve_trend=True, sigma=5, niters=5, timescale=3, bins=10, windows=20, normalize_and_split=True):
        self.method = validate_method(method, ['spline', 'lombscargle'])

        if design_matrix is None:
            design_matrix = self._normalize_and_split_design_matrix(self._basic_design_matrix(), self.breakindex)
        elif normalize_and_split:
            design_matrix = self._normalize_and_split_design_matrix(np.copy(design_matrix), self.breakindex)
        dm = np.copy(design_matrix)
        if cadence_mask is None:
            mask = np.ones(len(self.time), bool)
        else:
            mask = np.copy(cadence_mask)

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
        return
