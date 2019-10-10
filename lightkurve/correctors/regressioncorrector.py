"""Defines RegressionCorrector.
TO DO
-----
- Work when flux_err not available
- add regularization
"""
"""Defines RegressionCorrector."""
import inspect
import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from sklearn import linear_model

from .corrector import Corrector
from .designmatrix import DesignMatrix, DesignMatrixCollection
from ..lightcurve import LightCurve, MPLSTYLE
from ..utils import validate_method

__all__ = ['RegressionCorrector']

log = logging.getLogger(__name__)


class RegressionCorrector(Corrector):
    """Remove noise using linear regression against a design matrix.

    This method will use weighted linear least squares regression to find the
    parameter vector \beta which minimizes lc.flux - np.dot(X, beta)
    where X is a design matrix of regressors.

    Parameters
    ----------
    lc : `~lightkurve.lightcurve.LightCurve`
        The light curve that needs to be corrected.
    design_matrix : `~lightkurve.correctors.DesignMatrixCollection`
        A collection of one or more design matrices.  Each matrix in the
        collection must have a shape of (time, regressors).
        The columns contained in each matrix must be known to correlate with
        the signals or noise we want to remove from the light curve.
    model : str {'LinearRegression', 'Ridge', 'Lasso'}
            or a ~`sklearn.linear_model.base.LinearModel` instance
        Linear model to use.
    alpha : float
        Regularization strength. Only used if `model` equals 'Ridge' or 'Lasso'.
    """
    def __init__(self, lc, design_matrix_collection, model='LinearRegression', alpha=1.0):
        if isinstance(design_matrix_collection, DesignMatrix):
            design_matrix_collection = DesignMatrixCollection([design_matrix_collection])

        if isinstance(model, str):
            model = validate_method(model, ['linearregression', 'ridge', 'lasso'])
            if model == 'linearregression':
                # `fit_intercept=False` because we want the user to explicitely
                # include it as a column of ones in the design matrix
                model = linear_model.LinearRegression(fit_intercept=False)
            elif model == 'ridge':
                model = linear_model.Ridge(fit_intercept=False, alpha=alpha)
            elif model == 'lasso':
                model = linear_model.Lasso(fit_intercept=False, alpha=alpha)
        self.model = model

        # Validate user input
        if np.any([~np.isfinite(lc.time), ~np.isfinite(lc.flux)]):
            raise ValueError('Input light curve has NaNs in time or flux. '
                             'Please remove NaNs before correcting.')
        self.lc = lc
        design_matrix_collection._validate()
        self.X = design_matrix_collection

        # The following properties will be set when correct() is called:
        self.coefficients = None
        self.corrected_lc = None
        self.model_lc = None
        self.diagnostic_lightcurves = None

    def __repr__(self):
        return 'RegressionCorrector (LC: {})'.format(self.lc.targetid)

    def _fit_coefficients(self, cadence_mask=None, prior_mu=None, prior_sigma=None):
        """Fit the linear regression coefficients.

        Such that regression_model_flux = np.dot(X, beta) where X is the
        design matrix.

        Parameters
        ----------
        cadence_mask : np.ndarray of bool
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        coefficients : np.ndarray
            The best fit model coefficients to the data.
        """
        if prior_mu is not None:
            if len(prior_mu) != len(self.X.values.T):
                raise ValueError('Prior means must have shape {}'.format(len(self.X.values.T)))
        if prior_sigma is not None:
            if len(prior_sigma) != len(self.X.values.T):
                raise ValueError('Prior sigmas must have shape {}'.format(len(self.X.values.T)))

        # If prior_mu is specified, prior_sigma must be specified
        if not ((prior_mu is None) & (prior_sigma is None)) | ((prior_mu is not None) & (prior_sigma is not None)):
            raise ValueError("Please specify both `prior_mu` and `prior_sigma`")

        # Default cadence mask
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux), bool)
        X = self.X.values
        sigma_w_inv = np.dot(X[cadence_mask].T, X[cadence_mask] / self.lc.flux_err[cadence_mask, None]**2)
        if prior_sigma is not None:
            sigma_w_inv += 1/prior_sigma**2
        B = np.dot(X[cadence_mask].T, (self.lc.flux / self.lc.flux_err**2)[cadence_mask, None])
        if prior_sigma is not None:
            sigma_w_inv += prior_mu/prior_sigma**2
        w = np.linalg.solve(sigma_w_inv, B).T[0]
        w_err = np.linalg.inv(sigma_w_inv)
        return w, w_err

    def correct(self, cadence_mask=None, sigma=5, niters=5):
        """Find the best fit correction for the light curve.

        Parameters
        ----------
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers

        Returns
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Corrected light curve, with noise removed.
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.time), bool)
        else:
            cadence_mask = np.copy(cadence_mask)

        # Iterative sigma clipping
        for count in range(niters):
            coefficients, coefficients_err = self._fit_coefficients(cadence_mask=cadence_mask,
                                                            prior_mu=self.X.prior_mu,
                                                            prior_sigma=self.X.prior_sigma)
            residuals = self.lc.flux - np.dot(self.X.values, coefficients)
            cadence_mask &= ~sigma_clip(residuals, sigma=sigma).mask
            log.debug("correct(): iteration {}: clipped {} cadences"
                      "".format(count, cadence_mask.sum()))
        self.cadence_mask = cadence_mask
        self.coefficients = coefficients
        self.coefficients_err = coefficients_err

        model_flux = np.dot(self.X.values, coefficients)
        model_flux -= np.median(model_flux)
        samples = np.asarray([np.dot(self.X.values, np.random.multivariate_normal(coefficients, coefficients_err)) for idx in range(100)]).T
        model_err = np.abs(np.percentile(samples, [16, 84], axis=1) - np.median(samples, axis=1)[:, None].T).mean(axis=0)
        self.model_lc = LightCurve(self.lc.time, model_flux, model_err)
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        self.corrected_lc.flux_err = (self.lc.flux_err*2 + self.model_lc.flux_err**2)**0.5
        self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()
        return self.corrected_lc

    def _create_diagnostic_lightcurves(self):
        """Returns a dictionary containing all diagnostic light curves.

        The dictionary will provide a light curve for each matrix in the
        design matrix collection.
        """
        if self.coefficients is None:
            raise ValueError("you need to call correct() first")

        lcs = {}
        for idx, submatrix in enumerate(self.X.matrices):
            # What is the index of the first column for the submatrix?
            firstcol_idx = sum([m.shape[1] for m in self.X.matrices[:idx]])
            submatrix_coefficients = self.coefficients[firstcol_idx:firstcol_idx+submatrix.shape[1]]
#            submatrix_coefficients_err = self.coefficients_err[firstcol_idx:firstcol_idx+submatrix.shape[1], firstcol_idx:firstcol_idx+submatrix.shape[1]]
#            samples = np.asarray([np.dot(submatrix.values, np.random.multivariate_normal(submatrix_coefficients, submatrix_coefficients_err)) for idx in range(100)]).T
#            model_err = np.abs(np.percentile(samples, [16, 84], axis=1) - np.median(samples, axis=1)[:, None].T).mean(axis=0)
            model_flux = np.dot(submatrix.values, submatrix_coefficients)
            lcs[submatrix.name] = LightCurve(self.lc.time, model_flux - np.median(model_flux), label=submatrix.name)
        return lcs

    def _diagnostic_plot(self):
        """ Produce diagnostic plots to assess the effectiveness of the correction. """

        if not hasattr(self, 'corrected_lc'):
            raise ValueError('Please run `correct` method before trying to diagnose.')

        with plt.style.context(MPLSTYLE):
            fig, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
            ax = axs[0]
            self.lc.plot(ax=ax, normalize=False, label='Original', alpha=0.4)
            for key in self.diagnostic_lightcurves.keys():
                (self.diagnostic_lightcurves[key] + np.median(self.lc.flux)).plot(ax=ax)
            ax.set_xlabel('')
            ax = axs[1]
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label='Original')
            self.corrected_lc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x', s=10, label='Outliers', ax=ax)
            self.corrected_lc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return axs

    def diagnose(self):
        self._diagnostic_plot()
