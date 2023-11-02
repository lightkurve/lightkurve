"""Defines `RegressionCorrector` to solve large linear regression problems
with user-defined Gaussian priors in a fast, analytical way.
"""
import logging
import warnings

from astropy.stats import sigma_clip
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse, csr_matrix

from .corrector import Corrector
from .designmatrix import (
    DesignMatrix,
    DesignMatrixCollection,
    SparseDesignMatrix,
    SparseDesignMatrixCollection,
)
from ..lightcurve import LightCurve, MPLSTYLE


__all__ = ["RegressionCorrector"]


log = logging.getLogger(__name__)


class RegressionCorrector(Corrector):
    r"""Remove noise using linear regression against a `.DesignMatrix`.

    .. math::

        \newcommand{\y}{\mathbf{y}}
        \newcommand{\cov}{\boldsymbol\Sigma_\y}
        \newcommand{\w}{\mathbf{w}}
        \newcommand{\covw}{\boldsymbol\Sigma_\w}
        \newcommand{\muw}{\boldsymbol\mu_\w}
        \newcommand{\sigw}{\boldsymbol\sigma_\w}
        \newcommand{\varw}{\boldsymbol\sigma^2_\w}

    Given a column vector of data :math:`\y`
    and a design matrix of regressors :math:`X`,
    we will find the vector of coefficients :math:`\w`
    such that:

    .. math::

        \mathbf{y} = X\mathbf{w} + \mathrm{noise}

    We will assume that the model fits the data within Gaussian uncertainties:

    .. math::

        p(\y | \w) = \mathcal{N}(X\w, \cov)


    We make the regression robust by placing Gaussian priors on :math:`\w`:

    .. math::

        p(\w) = \mathcal{N}(\muw, \sigw)


    We can then find the maximum likelihood solution of the posterior
    distribution :math:`p(\w | \y) \propto p(\y | \w) p(\w)` by solving
    the matrix equation:

    .. math::

        \w = \covw (X^\\top \cov^{-1} \y + \\boldsymbol\sigma^{-2}_\w I \muw)

    Where :math:`\covw` is the covariance matrix of the coefficients:

    .. math::

        \covw^{-1} = (X^\\top \cov^{-1} X + \\boldsymbol\sigma^{-2}_\w I)


    Parameters
    ----------
    lc : `.LightCurve`
        The light curve that needs to be corrected.
    """

    def __init__(self, lc):
        # We don't accept NaN in time or flux.
        if np.any([~np.isfinite(lc.time.value), ~np.isfinite(lc.flux)]):
            raise ValueError(
                "Input light curve has NaNs in time or flux. "
                "Please remove NaNs before correction "
                "(e.g. using `lc = lc.remove_nans()`)."
            )
        # We don't accept NaN in flux_err, unless all values are NaN.
        if np.any(~np.isfinite(lc.flux_err)) and not np.all(~np.isfinite(lc.flux_err)):
            raise ValueError(
                "Input light curve has NaNs in `flux_err`. "
                "Please remove NaNs before correction "
                "(e.g. using `lc = lc.remove_nans()`)."
            )
        if np.any(lc.flux_err[np.isfinite(lc.flux_err)] <= 0):
            raise ValueError(
                "Input light curve contains flux uncertainties "
                "smaller than or equal to zero. Please remove "
                "these (e.g. using `lc = lc[lc.flux_err > 0]`)."
            )
        self.lc = lc

        # The following properties will be set when correct() is called.
        # We're setting them here so they do not throw value errors
        self.design_matrix_collection = None
        self.coefficients = None
        self.corrected_lc = None
        self.model_lc = None
        self.diagnostic_lightcurves = None

    def __repr__(self):
        return "RegressionCorrector (ID: {})".format(self.lc.targetid)

    @property
    def dmc(self):
        """Shorthand for self.design_matrix_collection."""
        return self.design_matrix_collection

    def _fit_coefficients(
        self, cadence_mask=None, prior_mu=None, prior_sigma=None, propagate_errors=False
    ):
        """Fit the linear regression coefficients.

        This function will solve a linear regression with Gaussian priors
        on the coefficients.

        Parameters
        ----------
        cadence_mask : np.ndarray of bool
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        coefficients : np.ndarray
            The best fit model coefficients to the data.
        """

        # If prior_mu is specified, prior_sigma must be specified
        if not ((prior_mu is None) & (prior_sigma is None)) | (
            (prior_mu is not None) & (prior_sigma is not None)
        ):
            raise ValueError("Please specify both `prior_mu` and `prior_sigma`")

        # Default cadence mask
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux.value), bool)

        # If flux errors are not all finite numbers, then default to array of ones
        if np.all(~np.isfinite(self.lc.flux_err.value)):
            flux_err = np.ones(cadence_mask.sum())
        else:
            flux_err = self.lc.flux_err.value[cadence_mask]

        # Retrieve the design matrix (X) as a numpy array
        X = self.dmc.X[cadence_mask]
        if isinstance(X, np.ndarray):
            # Compute `X^T cov^-1 X + 1/prior_sigma^2`
            sigma_w_inv = X.T.dot(X / flux_err[:, None] ** 2)
            # Compute `X^T cov^-1 y + prior_mu/prior_sigma^2`
            B = np.dot(X.T, self.lc.flux.value[cadence_mask] / flux_err ** 2)

        elif issparse(X):
            sigma_f_inv = csr_matrix(1 / flux_err[:, None] ** 2)
            # Compute `X^T cov^-1 X + 1/prior_sigma^2`
            sigma_w_inv = X.T.dot(X.multiply(sigma_f_inv))
            # Compute `X^T cov^-1 y + prior_mu/prior_sigma^2`
            B = X.T.dot((self.lc.flux[cadence_mask] / flux_err ** 2))
            sigma_w_inv = sigma_w_inv.toarray()

        if prior_sigma is not None:
            sigma_w_inv = sigma_w_inv + np.diag(1.0 / prior_sigma ** 2)
            B = B + (prior_mu / prior_sigma ** 2)

        # Solve for weights w
        w = np.linalg.solve(sigma_w_inv, B).T
        if propagate_errors:
            w_err = np.linalg.inv(sigma_w_inv)
        else:
            w_err = np.zeros(len(w)) * np.nan

        return w, w_err

    def correct(
        self,
        design_matrix_collection,
        cadence_mask=None,
        sigma=5,
        niters=5,
        propagate_errors=False,
    ):
        """Find the best fit correction for the light curve.

        Parameters
        ----------
        design_matrix_collection : `.DesignMatrix` or `.DesignMatrixCollection`
            One or more design matrices.  Each matrix must have a shape of
            (time, regressors). The columns contained in each matrix must be
            known to correlate with additive noise components we want to remove
            from the light curve.
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        propagate_errors : bool (default False)
            Whether to propagate the uncertainties from the regression. Default is False.
            Setting to True will increase run time, but will sample from multivariate normal
            distribution of weights.

        Returns
        -------
        corrected_lc : `.LightCurve`
            Corrected light curve, with noise removed.
        """
        if not isinstance(design_matrix_collection, DesignMatrixCollection):
            if isinstance(design_matrix_collection, SparseDesignMatrix):
                design_matrix_collection = SparseDesignMatrixCollection(
                    [design_matrix_collection]
                )
            elif isinstance(design_matrix_collection, DesignMatrix):
                design_matrix_collection = DesignMatrixCollection(
                    [design_matrix_collection]
                )

        # Validate the design matrix. Emits a warning if the matrix has low rank.
        design_matrix_collection.validate()
        self.design_matrix_collection = design_matrix_collection

        if cadence_mask is None:
            self.cadence_mask = np.ones(len(self.lc.time), bool)
        else:
            self.cadence_mask = cadence_mask

        # Create an outlier mask using iterative sigma clipping
        self.outlier_mask = np.zeros_like(self.cadence_mask)
        for count in range(niters):
            tmp_cadence_mask = self.cadence_mask & ~self.outlier_mask
            coefficients, coefficients_err = self._fit_coefficients(
                cadence_mask=tmp_cadence_mask,
                prior_mu=self.dmc.prior_mu,
                prior_sigma=self.dmc.prior_sigma,
                propagate_errors=propagate_errors,
            )
            model = np.ma.masked_array(
                data=self.dmc.X.dot(coefficients), mask=~tmp_cadence_mask
            )
            model = u.Quantity(model, unit=self.lc.flux.unit)
            residuals = self.lc.flux - model
            if isinstance(residuals, Masked):
                # Workaround for https://github.com/astropy/astropy/issues/14360
                # in passing MaskedQuantity to sigma_clip, by converting it to Quantity.
                # We explicitly fill masked values with `np.nan` here to ensure they are masked during sigma clipping.
                # To handle unlikely edge case, convert int to float to ensure filing `np.nan` work.
                # The conversion is acceptable because only the mask of the sigma_clip() result is used.
                if np.issubdtype(residuals.dtype, np.int_):
                    residuals = residuals.astype(float)
                residuals = residuals.filled(np.nan)
            with warnings.catch_warnings():  # Ignore warnings due to NaNs
                warnings.simplefilter("ignore", AstropyUserWarning)
                self.outlier_mask |= sigma_clip(residuals, sigma=sigma).mask
            log.debug(
                "correct(): iteration {}: clipped {} cadences"
                "".format(count, self.outlier_mask.sum())
            )

        self.coefficients = coefficients
        self.coefficients_err = coefficients_err

        model_flux = self.dmc.X.dot(coefficients)
        model_flux -= np.median(model_flux)
        if propagate_errors:
            with warnings.catch_warnings():
                # ignore "RuntimeWarning: covariance is not symmetric positive-semidefinite."
                warnings.simplefilter("ignore", RuntimeWarning)
                samples = np.asarray(
                    [
                        self.dmc.X.dot(
                            np.random.multivariate_normal(
                                coefficients, coefficients_err
                            )
                        )
                        for idx in range(100)
                    ]
                ).T
            model_err = np.abs(
                np.percentile(samples, [16, 84], axis=1)
                - np.median(samples, axis=1)[:, None].T
            ).mean(axis=0)
        else:
            model_err = np.zeros(len(model_flux))
        self.model_lc = LightCurve(
            time=self.lc.time,
            flux=u.Quantity(model_flux, unit=self.lc.flux.unit),
            flux_err=u.Quantity(model_err, unit=self.lc.flux.unit),
        )
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        self.corrected_lc.flux_err = (self.lc.flux_err ** 2 + model_err ** 2) ** 0.5
        self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()
        return self.corrected_lc

    def _create_diagnostic_lightcurves(self):
        """Returns a dictionary containing all diagnostic light curves.

        The dictionary will provide a light curve for each matrix in the
        design matrix collection.
        """
        if self.coefficients is None:
            raise ValueError("you need to call `correct()` first")

        lcs = {}
        for idx, submatrix in enumerate(self.dmc.matrices):
            # What is the index of the first column for the submatrix?
            firstcol_idx = sum([m.shape[1] for m in self.dmc.matrices[:idx]])
            submatrix_coefficients = self.coefficients[
                firstcol_idx : firstcol_idx + submatrix.shape[1]
            ]
            # submatrix_coefficients_err = self.coefficients_err[firstcol_idx:firstcol_idx+submatrix.shape[1], firstcol_idx:firstcol_idx+submatrix.shape[1]]
            # samples = np.asarray([np.dot(submatrix.values, np.random.multivariate_normal(submatrix_coefficients, submatrix_coefficients_err)) for idx in range(100)]).T
            # model_err = np.abs(np.percentile(samples, [16, 84], axis=1) - np.median(samples, axis=1)[:, None].T).mean(axis=0)
            model_flux = u.Quantity(
                submatrix.X.dot(submatrix_coefficients), unit=self.lc.flux.unit
            )
            model_flux_err = u.Quantity(
                np.zeros(len(model_flux)), unit=self.lc.flux.unit
            )
            lcs[submatrix.name] = LightCurve(
                time=self.lc.time,
                flux=model_flux,
                flux_err=model_flux_err,
                label=submatrix.name,
            )
        return lcs

    def _diagnostic_plot(self):
        """Produce diagnostic plots to assess the effectiveness of the correction.

        Note: We need a hidden function so that other correctors can alter the plot.
        """
        if not hasattr(self, "corrected_lc"):
            raise ValueError(
                "Please call the `correct()` method before trying to diagnose."
            )

        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(2, figsize=(10, 6), sharex=True)
            ax = axs[0]
            self.lc.plot(ax=ax, normalize=False, label="original", alpha=0.4)
            for key in self.diagnostic_lightcurves.keys():
                (
                    self.diagnostic_lightcurves[key]
                    - np.median(self.diagnostic_lightcurves[key].flux)
                    + np.median(self.lc.flux)
                ).plot(ax=ax)
            ax.set_xlabel("")
            ax = axs[1]
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label="original")
            self.corrected_lc[self.outlier_mask].scatter(
                normalize=False, c="r", marker="x", s=10, label="outlier_mask", ax=ax
            )
            self.corrected_lc[~self.cadence_mask].scatter(
                normalize=False,
                c="dodgerblue",
                marker="x",
                s=10,
                label="~cadence_mask",
                ax=ax,
            )
            self.corrected_lc.plot(normalize=False, label="corrected", ax=ax, c="k")
        return axs

    def diagnose(self):
        """Returns diagnostic plots to assess the most recent call to `correct()`.

        If `correct()` has not yet been called, a ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        return self._diagnostic_plot()

    def diagnose_priors(self):
        """Returns a diagnostic plot visualizing how the best-fit coefficients
        compare against the priors.

        The method will show the results obtained during the most recent call
        to `correct()`.  If `correct()` has not yet been called, a
        ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if not hasattr(self, "corrected_lc"):
            raise ValueError(
                "Please call the `correct()` method before trying to diagnose."
            )

        names = [dm.name for dm in self.dmc]
        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(
                1, len(names), figsize=(len(names) * 4, 4), sharey=True
            )
            if not hasattr(axs, "__iter__"):
                axs = [axs]
            for idx, ax, X in zip(range(len(names)), axs, self.dmc):
                X.plot_priors(ax=ax)
                firstcol_idx = sum([m.shape[1] for m in self.dmc.matrices[:idx]])
                submatrix_coefficients = self.coefficients[
                    firstcol_idx : firstcol_idx + X.shape[1]
                ]
                [ax.axvline(s, color="red", zorder=-1) for s in submatrix_coefficients]
        return axs
