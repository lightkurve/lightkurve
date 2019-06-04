"""Defines PyMCPLDCorrector.

TODO Now
--------
* [DONE] Implement diagnostic plots, `corr.plot_design_matrix()` and `corr.plot_diagnostics()`.
* Port the existing suite of PLDCorrector tests.
* Fix the units of corrected_flux, corrected_flux_without_star, etc.
  In Lightkurve we currently add back the mean of the model.
* Document the meaning of logs2, logsigma (amplitude of variability),
  and logrho (timescale of variability), and make sure their
  prior values are configurable.

TODO Before release
-------------------
* Rename PyMCPLDCorrector as PLDCorrector.
* The design matrix can be improved by rejecting pixels which are saturated,
  and including the collapsed sums of their CCD columns instead.
* Add pymc & exoplanet & theano to Lightkurve's dependencies or treat it properly
  as an optional import. [Geert]
* Set PyMC verbosity to match the lightkurve.log.level. [Geert]

Future enhancements
-------------------
* In `create_first_order_matrix`, we need to explain why we are dividing each
  cadence by the sum of all pixels. We think this is explained in the EVEREST paper,
  i.e. is is likely because PLD optimizes towards a flat light curve. We need
  to find the right words to explain this and refer to papers where possible.
* It is not clear whether the inclusion of a column vector of ones in the
  design matrix is necessary for numerical stability.
"""
from __future__ import division, print_function

import logging
from itertools import combinations_with_replacement as multichoose

import numpy as np
import pymc3 as pm
import exoplanet as xo
import theano.tensor as tt
import matplotlib.pyplot as plt

from .. import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['PyMCPLDCorrector']


class PyMCPLDCorrector(object):
    r"""Implements the Pixel Level Decorrelation (PLD) systematics removal method.

        Pixel Level Decorrelation (PLD) was developed by [1]_ to remove
        systematic noise caused by spacecraft jitter for the Spitzer
        Space Telescope. It was adapted to K2 data by [2]_ and [3]_
        for the EVEREST pipeline [4]_.

        For a detailed description and implementation of PLD, please refer to
        these references. Lightkurve provides a reference implementation
        of PLD that is less sophisticated than EVEREST, but is suitable
        for quick-look analyses and detrending experiments.

        Our simple implementation of PLD is performed by first calculating the
        noise model for each cadence in time. This function goes up to arbitrary
        order, and is represented by

        .. math::

            m_i = \alpha + \beta t_i + \gamma t_i^2 + \sum_l a_l \frac{f_{il}}{\sum_k f_{ik}} + \sum_l \sum_m b_{lm} \frac{f_{il}f_{im}}{\left( \sum_k f_{ik} \right)^2} + ...
        where

          - :math:`m_i` is the noise model at time :math:`t_i`
          - :math:`f_{il}` is the flux in the :math:`l^\text{th}` pixel at time :math:`t_i`
          - :math:`a_l` is the first-order PLD coefficient on the linear term
          - :math:`b_{lm}` is the second-order PLD coefficient on the :math:`l^\text{th}`,
            :math:`m^\text{th}` pixel pair
          - :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are the
            Gaussian Process terms applied to capture long-period variability.

        We perform Principal Component Analysis (PCA) to reduce the number of
        vectors in our final model to limit the set to best capture instrumental
        noise. With a PCA-reduced set of vectors, we can construct a design matrix
        containing fractional pixel fluxes.

        To solve for the PLD model, we need to minimize the difference squared

        .. math::

            \chi^2 = \sum_i \frac{(y_i - m_i)^2}{\sigma_i^2},

        where :math:`y_i` is the observed flux value at time :math:`t_i`, by solving

        .. math::

            \frac{\partial \chi^2}{\partial a_l} = 0.

    References
    ----------
    .. [1] Deming et al. (2015), ads:2015ApJ...805..132D.
        (arXiv:1411.7404)
    .. [2] Luger et al. (2016), ads:2016AJ....152..100L
        (arXiv:1607.00524)
    .. [3] Luger et al. (2018), ads:2018AJ....156...99L
        (arXiv:1702.05488)
    .. [4] EVEREST pipeline webpage, https://rodluger.github.io/everest
    """
    def __init__(self, tpf, aperture_mask=None, pld_aperture_mask=None):
        # Input validation: parse the aperture masks to accept strings etc.
        self.aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        self.pld_aperture_mask = tpf._parse_aperture_mask(pld_aperture_mask)
        # Generate raw flux light curve from desired pixels
        raw_lc = tpf.to_lightcurve(aperture_mask=self.aperture_mask)
        # It is critical to remove all NaNs or the linear algebra below will crash
        self.raw_lc, self.nan_mask = raw_lc.remove_nans(return_mask=True)
        self.tpf = tpf[~self.nan_mask]

    def create_first_order_matrix(self):
        """Returns a matrix which encodes the fractional pixel fluxes as a function
        of cadence (row) and pixel (column). As such, the method returns a
        2D matrix with shape (n_cadences, n_pixels_in_pld_mask).

        This matrix will form the basis of the PLD regressor design matrix
        and is often called the first order component.

        The matrix returned is guaranteed to be free of NaN values.

        Returns
        -------
        matrix : numpy array
            First order PLD design matrix.
        """
        # Re-arrange the cube of flux values observed in a user-specified mask
        # into a 2D matrix of shape (n_cadences, n_pixels_in_mask).
        # Note that Theano appears to require 64-bit floats.
        matrix = np.asarray(self.tpf.flux[:, self.pld_aperture_mask], np.float64)
        assert matrix.shape == (len(self.raw_lc.time), self.pld_aperture_mask.sum())
        # Remove all NaN or Inf values
        matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
        # To ensure that each column contains the fractional pixel flux,
        # we divide by the sum of all pixels in the same cadence.
        # The reason for doing this is described in Section 2 of Luger et al. (2016).
        matrix = matrix / np.sum(matrix, axis=-1)[:, None]
        # If we return matrix at this point, theano will raise a "dimension mismatch".
        # The origin of this bug is not understood, but copying the matrix
        # into a new one as shown below circumvents it:
        result = np.empty((matrix.shape[0], matrix.shape[1]))
        result[:, :] = matrix[:, :]
        return result

    def create_design_matrix(self, pld_order=1, n_pca_terms=10, include_column_of_ones=False):
        """Returns a matrix designed to contain suitable regressors for the
        systematics noise model.

        The design matrix contains one row for each cadence (i.e. moment in time)
        and one column for each regressor that we wish to use to predict the
        systematic noise in a given cadence.

        The columns (i.e. regressors) included in the design matrix are:
        * One column for each pixel in the PLD aperture mask.  Each column
          contains the flux values observed by that pixel over time.  This is
          also known as the first order component.
        * Columns derived from the products of all combinations of pixel values
          in the aperture mask. However, rather than including a column for each
          combination, we perform dimensionality reduction (PCA) and include a
          smaller number of PCA terms, i.e. the number of columns is
          n_pca_terms*(pld_order-1).  This is also known as the higher order
          components.
        * Optionally, a single column of ones for numerical stability.

        Thus, the shape of the design matrix will be
        (n_cadences, n_pld_mask_pixels + n_pca_terms*(pld_order-1) + include_column_of_ones)

        Returns
        -------
        design_matrix : 2D numpy array
            See description above.
        """
        # We use an optional dependency for very fast PCA (fbpca), but if the
        # import fails we will fall back on using the slower `np.linalg.svd`.
        use_fbpca = True
        try:
            from fbpca import pca
        except ImportError:
            use_fbpca = False
            log.warning("PLD systematics correction will run faster if the "
                        "optional `fbpca` package is installed "
                        "(`pip install fbpca`).")

        matrix_sections = []  # list to hold the design matrix components
        first_order_matrix = self.create_first_order_matrix()

        # Input validation: n_pca_terms cannot be larger than the number of cadences
        n_cadences = len(first_order_matrix)
        if n_pca_terms > n_cadences:
            log.warning("`n_pca_terms` ({}) cannot be larger than the number of cadences ({});"
                        "using n_pca_terms={}".format(n_pca_terms, n_cadences, n_cadences))
            n_pca_terms = n_cadences

        # The original EVEREST paper includes a column vector of ones in the
        # design matrix to improve the numerical stability (see Luger et al.);
        # it is unclear whether this is necessary, so this is an optional step
        # for now.
        if include_column_of_ones:
            matrix_sections.append([np.ones((len(first_order_matrix), 1))])

        # Add the first order matrix
        matrix_sections.append(first_order_matrix)

        # Add the higher order PLD design matrix columns
        for order in range(2, pld_order + 1):
            # Take the product of all combinations of pixels; order=2 will
            # multiply all pairs of pixels, order=3 will multiple triples, etc.
            matrix = np.product(list(multichoose(first_order_matrix.T, order)), axis=1).T
            # This product matrix becomes very big very quickly, so we reduce
            # its dimensionality using PCA.
            if use_fbpca:  # fast mode
                components, _, _ = pca(matrix, n_pca_terms)
            else:  # slow mode
                components, _, _ = np.linalg.svd(matrix)
            section = components[:, :n_pca_terms]
            matrix_sections.append(section)

        return np.concatenate(matrix_sections, axis=1)

    def create_pymc_model(self, design_matrix=None, cadence_mask=None, gp_timescale=150, **kwargs):
        """Returns a PYMC3 model.

        Parameters
        ----------
        design_matrix : np.ndarray
            Matrix of shape (n_cadences, n_regressors) used to create the
            motion model.
        cadence_mask : np.ndarray
            Boolean array to mask cadences. Cadences that are False will be excluded
            from the model fit

        Returns
        -------
        model : pymc3.model.Model
            A pymc3 model
        """
        if design_matrix is None:
            design_matrix = self.create_design_matrix(**kwargs)
        if cadence_mask is None:
            cadence_mask = np.zeros(len(self.raw_lc.time), dtype=bool)

        # Theano raises an `AsTensorError` ("Cannot convert to TensorType")
        # if the floats are not in 64-bit format, so we convert them here:
        lc_time = np.asarray(self.raw_lc.time, np.float64)
        lc_flux = np.asarray(self.raw_lc.flux, np.float64)
        lc_flux_err = np.asarray(self.raw_lc.flux_err, np.float64)

        # Covariance matrix diagonal
        diag = lc_flux_err**2

        # The cadence mask is applied by inflating the uncertainties in the covariance matrix;
        # this is because celerite will run much faster if it is able to predict
        # data for cadences that have been fed into the model.
        diag[cadence_mask] += 1e12

        with pm.Model() as model:
            # Create a Gaussian Process to model the long-term stellar variability
            logs2 = pm.Normal("logs2", mu=np.log(np.var(lc_flux)), sd=4)
            logsigma = pm.Normal("logsigma", mu=np.log(np.std(lc_flux)), sd=4)
            logrho = pm.Normal("logrho", mu=np.log(gp_timescale), sd=4)
            kernel = xo.gp.terms.Matern32Term(log_sigma=logsigma, log_rho=logrho)
            gp = xo.gp.GP(kernel, lc_time, diag + tt.exp(logs2))

            # The motion model regresses against the design matrix
            A = tt.dot(design_matrix.T, gp.apply_inverse(design_matrix))
            B = tt.dot(design_matrix.T, gp.apply_inverse(lc_flux[:, None]))
            weights = tt.slinalg.solve(A, B)
            motion_model = pm.Deterministic("motion_model", tt.dot(design_matrix, weights)[:, 0])
            pm.Deterministic("weights", weights)

            # Likelihood to optimize
            pm.Potential("obs", gp.log_likelihood(lc_flux - motion_model))

            # Add deterministic variables for easy of use
            star_model = gp.predict()
            pm.Deterministic("lc_time", tt.as_tensor_variable(lc_time))
            pm.Deterministic("lc_flux", tt.as_tensor_variable(lc_flux))
            pm.Deterministic("star_model", star_model)
            pm.Deterministic("corrected_flux", lc_flux - motion_model)
            pm.Deterministic("corrected_flux_without_star", lc_flux - motion_model - star_model)

        return model

    def optimize(self, model=None, start=None, **kwargs):
        """Returns the maximum likelihood solution.

        Returns
        -------
        map_soln : dict
            Maximum likelihood values.
        """
        if model is None:
            model = self.create_pymc_model(**kwargs)
        if start is None:
            start = model.test_point

        with model:
            map_soln = xo.optimize(start=start, vars=[model.logsigma])
            map_soln = xo.optimize(start=start, vars=[model.logrho, model.logsigma])
            map_soln = xo.optimize(start=start, vars=[model.logsigma])
            map_soln = xo.optimize(start=start, vars=[model.logrho, model.logsigma])
            map_soln = xo.optimize(start=map_soln, vars=[model.logs2])
            map_soln = xo.optimize(start=map_soln, vars=[model.logrho, model.logsigma, model.logs2])
        return map_soln

    def sample(self, model=None, start=None, ndraws=1000):
        """Sample the systematics correction model."""
        if model is None:
            model = self.create_pymc_model()
        if start is None:
            start = self.optimize()
        sampler = xo.PyMC3Sampler()
        with model:
            burnin = sampler.tune(tune=np.max([int(ndraws*0.3), 150]),
                                  start=start,
                                  step_kwargs=dict(target_accept=0.9),
                                  chains=4)
            trace = sampler.sample(draws=ndraws, chains=4)
        return trace

    def correct(self, sample=False, **kwargs):
        """Returns a systematics-corrected light curve."""
        corrected_lc = self.raw_lc.copy()
        if sample:
            raise NotImplementedError("We've been slacking")
            #call self.sample(**kwargs)
            #set corrected_flux to mean of the sampled corrected light curves
            #add sample errors in quadrature to existing flux_err?
        else:
            sol = self.optimize(**kwargs)
            corrected_lc.flux = sol['corrected_flux']
        return corrected_lc

    def plot_diagnostics(self, solution=None, **kwargs):
        """Plots a series of useful figures to help understand the noise removal
        process.

        IDEA: Add indicator for masked cadences in plotted light curves."""
        if solution is None:
            solution = self.optimize(**kwargs)

        # until we fix units on corrected light curves, we need to add back the
        # mean of the subtracted motion model when plotting
        mean = np.mean(solution["motion_model"])

        # Set up subplots with lightkurve plot style
        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(8.485, 10))
            # Plot the raw light curve with the noise model overlaid
            ax[0].plot(solution["lc_time"], solution["lc_flux"], "r.", alpha=0.3, label="Raw Flux")
            ax[0].plot(solution["lc_time"], solution["corrected_flux"]+mean, "k.", label="Corrected Flux")
            ax[0].set_ylabel("Flux [e$^-$s$^{-1}$]")
            ax[0].legend(loc="best")

            # Plot the raw light curve with motion noise model overlaid
            ax[1].plot(solution["lc_time"], solution["lc_flux"], "r.", alpha=0.3, label="Raw Flux")
            ax[1].plot(solution["lc_time"], solution["motion_model"], "k.", label="Motion Model")
            ax[1].set_ylabel("Flux [e$^-$s$^{-1}$]")
            ax[1].legend(loc="best")

            # Plot the raw light curve with GP overlaid
            ax[2].plot(solution["lc_time"], solution["lc_flux"], "r.", alpha=0.3, label="Raw Flux")
            ax[2].plot(solution["lc_time"], solution["star_model"]+mean, "k", label="Stellar Model")
            ax[2].set_ylabel("Flux [e$^-$s$^{-1}$]")
            ax[2].legend(loc="best")

            fig.tight_layout()
            plt.xlabel("Time - 2454833 [BKJD days]")

    def plot_design_matrix(self, design_matrix=None, **kwargs):
        """Plots the design matrix.
        I'm not convinced that this is a valuable diagnostic..."""
        if design_matrix is None:
            design_matrix = self.create_design_matrix(**kwargs)

        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots(1, figsize=(5,5))
            ax.imshow(design_matrix, aspect='auto')
            ax.set_ylabel('Cadence Number')
            ax.set_xlabel('Regressors')
