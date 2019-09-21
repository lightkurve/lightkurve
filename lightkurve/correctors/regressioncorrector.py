"""Defines RegressionCorrector."""
import logging

import numpy as np
from astropy.stats import sigma_clip

from .corrector import Corrector
from .designmatrix import DesignMatrix, DesignMatrixCollection
from ..lightcurve import LightCurve

__all__ = ['RegressionCorrector']

log = logging.getLogger(__name__)


class RegressionCorrector(Corrector):
    """Remove noise using a linear fit of the light curve against a design matrix of regressors.

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
    """
    def __init__(self, lc, design_matrix_collection):
        if isinstance(design_matrix_collection, DesignMatrix):
            design_matrix_collection = DesignMatrixCollection([design_matrix_collection])
        # Validate user input
        if np.any([~np.isfinite(lc.time), ~np.isfinite(lc.flux), ~np.isfinite(lc.flux_err)]):
            raise ValueError('Input light curve has NaNs in time, flux, and/or flux_err. '
                             'Please remove NaNs before correcting.')
        self.lc = lc
        design_matrix_collection._validate()
        self.X = design_matrix_collection

        # The following properties will be set when correct() is called:
        self.coefficients = None
        self.corrected_lc = None
        self.model_lc = None
        self.diagnostic_lightcurves = None

    def _fit_coefficients(self, cadence_mask=None):
        """Fit the linear regression coefficients.

        Such that regression_model_flux = np.dot(X, beta) where X is the
        design matrix.

        Parameters
        ----------
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        coefficients : np.ndarray
            The best fit model coefficients to the data.
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux), bool)
        dm = self.X.values
        flux = self.lc.flux[cadence_mask]
        flux_weights = self.lc.flux_err[cadence_mask]**2

        A = np.dot(dm[cadence_mask].T, dm[cadence_mask] / flux_weights[:, None])
        B = np.dot(dm[cadence_mask].T, flux / flux_weights)
        coefficients = np.linalg.solve(A, B)
        return coefficients

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
            coefficients = self._fit_coefficients(cadence_mask=cadence_mask)
            residuals = self.lc.flux - np.dot(self.X.values, coefficients)
            cadence_mask &= ~sigma_clip(residuals, sigma=sigma).mask
            log.debug("correct(): iteration {}: clipped {} cadences"
                      "".format(count, cadence_mask.sum()))
        self.cadence_mask = cadence_mask
        self.coefficients = coefficients

        model_flux = np.dot(self.X.values, coefficients)
        model_flux -= np.median(model_flux)
        self.model_lc = LightCurve(self.lc.time, model_flux)
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
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
            model_flux = np.dot(submatrix.values, submatrix_coefficients)
            lcs[submatrix.name] = LightCurve(self.lc.time, model_flux - np.median(model_flux), label=submatrix.name)
        return lcs

    def diagnose(self):
        """ Produce diagnostic plots to assess the effectiveness of the correction. """


        # SHOULD NOT BE CALLABLE BEFORE CORRECT
        ax = self.lc.plot(normalize=False, label='Original', alpha=0.4)
        for key in self.diagnostic_lightcurves.keys():
            (self.diagnostic_lightcurves[key] + np.median(self.lc.flux)).plot(ax=ax)

        ax = self.lc.plot(normalize=False, alpha=0.2, label='Original')
        self.corrected_lc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x', s=10, label='Outliers', ax=ax)
        self.corrected_lc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return
