"""Defines PLDCorrector
"""
from __future__ import division, print_function

import logging
from itertools import combinations_with_replacement as multichoose

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from .corrector import Corrector
from .gpcorrector import GPCorrector
from .. import MPLSTYLE
from ..utils import LightkurveWarning, suppress_stdout

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']

class PLDCorrector(Corrector):
    r"""Implements the Pixel Level Decorrelation (PLD) systematics removal method.
        Pixel Level Decorrelation (PLD) was developed by [1]_ to remove
        systematic noise caused by spacecraft jitter for the Spitzer
        Space Telescope. It was adapted to K2 data by [2]_ and [3]_
        for the EVEREST pipeline [4]_.

        For a detailed description and implementation of PLD, please refer to
        these references. Lightkurve provides a reference implementation
        of PLD that is less sophisticated than EVEREST, but is suitable
        for quick-look analyses and detrending experiments.

        Background
        ----------
        Our implementation of PLD is performed by first calculating the noise
        model for each cadence in time. This function goes up to arbitrary
        order, and is represented by
        .. math::

            m_i = \sum_l a_l \frac{f_{il}}{\sum_k f_{ik}} + \sum_l \sum_m b_{lm} \frac{f_{il}f_{im}}{\left( \sum_k f_{ik} \right)^2} + ...

        where

          - :math:`m_i` is the noise model at time :math:`t_i`
          - :math:`f_{il}` is the flux in the :math:`l^\text{th}` pixel at time :math:`t_i`
          - :math:`a_l` is the first-order PLD coefficient on the linear term
          - :math:`b_{lm}` is the second-order PLD coefficient on the :math:`l^\text{th}`,
            :math:`m^\text{th}` pixel pair

         We perform Principal Component Analysis (PCA) to reduce the number of
        vectors in our final model to limit the set to best capture instrumental
        noise. With a PCA-reduced set of vectors, we can construct a design matrix
        containing fractional pixel fluxes.

        To capture long-term variability, we simultaneously fit a Gaussian Process
        model ([5]_) to the underlying stellar signal.

        To solve for the PLD model, we need to minimize the difference squared

        .. math::

            \chi^2 = \sum_i \frac{(y_i - m_i)^2}{\sigma_i^2},

        where :math:`y_i` is the observed flux value at time :math:`t_i`, by solving

        .. math::

            \frac{\partial \chi^2}{\partial a_l} = 0.

    Examples
    --------
    Download the pixel data for GJ 9827 and obtain a PLD-corrected light curve:

    >>> import lightkurve as lk
    >>> tpf = lk.search_targetpixelfile("GJ9827").download() # doctest: +SKIP
    >>> corrector = lk.PLDCorrector(tpf) # doctest: +SKIP
    >>> lc = corrector.correct() # doctest: +SKIP
    >>> lc.plot() # doctest: +SKIP

    However, the above example will over-fit the small transits!
    It is necessary to mask the transits using `corrector.correct(cadence_mask=...)`.

    References
    ----------
    .. [1] Deming et al. (2015), ads:2015ApJ...805..132D.
        (arXiv:1411.7404)
    .. [2] Luger et al. (2016), ads:2016AJ....152..100L
        (arXiv:1607.00524)
    .. [3] Luger et al. (2018), ads:2018AJ....156...99L
        (arXiv:1702.05488)
    .. [4] EVEREST pipeline webpage, https://rodluger.github.io/everest
    .. [5] Celerite documentation, https://celerite.readthedocs.io/en/stable/

    Parameters
    ----------
    tpf : `TargetPixelFile` object
        The pixel data from which a light curve will be extracted.
    aperture_mask : 2D boolean array or str
        The pixel aperture mask that will be used to extract the raw light curve.
    design_matrix_aperture_mask : 2D boolean array or str
        The pixel aperture mask that will be used to create the regression matrix
        (i.e. the design matrix) used to model the systematics.  If `None`,
        then the `aperture_mask` value will be used.
    """

    def __init__(self, tpf, aperture_mask=None, design_matrix_aperture_mask='all'):
        # Use pipeline_mask by default
        if aperture_mask is None:
            aperture_mask = tpf.pipeline_mask
            if np.sum(aperture_mask) == 0:
                log.warning('No pixels in pipeline aperture mask. Using a threshold mask instead.')
                aperture_mask = 'threshold'
        # Input validation: parse the aperture masks to accept strings etc.
        self.aperture_mask = tpf._parse_aperture_mask(aperture_mask)
        self.design_matrix_aperture_mask = tpf._parse_aperture_mask(design_matrix_aperture_mask)
        # Generate raw flux light curve from desired pixels
        raw_lc = tpf.to_lightcurve(aperture_mask=self.aperture_mask)
        # It is critical to remove all cadences with NaNs or the linear algebra below will crash
        self.raw_lc, self.nan_mask = raw_lc.remove_nans(return_mask=True)
        self.tpf = tpf[~self.nan_mask]
        self.optimized = False

    def _create_first_order_matrix(self, normalize=True):
        """Returns a matrix which encodes the fractional pixel fluxes as a function
        of cadence (row) and pixel (column). As such, the method returns a
        2D matrix with shape (n_cadences, n_pixels_in_pld_mask).
        This matrix will form the basis of the PLD regressor design matrix
        and is often called the first order component. The matrix returned
        here is guaranteed to be free of NaN values.
        Returns
        -------
        matrix : numpy array
            First order PLD design matrix.
        """
        # Re-arrange the cube of flux values observed in a user-specified mask
        # into a 2D matrix of shape (n_cadences, n_pixels_in_mask).
        matrix = np.asarray(self.tpf.flux[:, self.design_matrix_aperture_mask])
        assert matrix.shape == (len(self.raw_lc.time), self.design_matrix_aperture_mask.sum())
        # Remove all NaN or Inf values
        matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
        # To ensure that each column contains the fractional pixel flux,
        # we divide by the sum of all pixels in the same cadence.
        # This is an important step, as explained in Section 2 of Luger et al. (2016).
        if normalize:
            matrix = matrix / np.sum(matrix, axis=-1)[:, None]

        return matrix

    def create_design_matrix(self, pld_order=2, n_pca_terms=10):
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

        Thus, the shape of the design matrix will be
        (n_cadences, n_pld_mask_pixels + n_pca_terms*(pld_order-1))

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
        first_order_matrix = self._create_first_order_matrix()

        # Input validation: n_pca_terms cannot be larger than the number of regressors (pixels)
        n_pixels = len(first_order_matrix.T)
        if n_pca_terms > n_pixels:
            log.warning("`n_pca_terms` ({}) cannot be larger than the number of pixels ({});"
                        "using n_pca_terms={}".format(n_pca_terms, n_pixels, n_pixels))
            n_pca_terms = n_pixels

        # Get the normalization matrix
        norm = np.sum(self._create_first_order_matrix(normalize=False), axis=1)[:, None]

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
            # Normalize the higher order components
            section = section / norm**order
            matrix_sections.append(section)
        if use_fbpca:  # fast mode
            first_order_matrix, _, _ = pca(first_order_matrix, n_pca_terms)
        else:  # slow mode
            first_order_matrix, _, _ = np.linalg.svd(first_order_matrix)[:, :n_pca_terms]

        # If we return matrix at this point, theano will raise a "dimension mismatch".
        # The origin of this bug is not understood, but copying the matrix
        # into a new one as shown below circumvents it:
        result = np.empty((first_order_matrix.shape[0], first_order_matrix.shape[1]))
        result[:, :] = first_order_matrix[:, :]

        # Add the first order matrix
        matrix_sections.insert(0, first_order_matrix)
        design_matrix = np.concatenate(matrix_sections, axis=1)

        # No columns in the design matrix should be NaN
        assert np.isfinite(design_matrix).any()

        return design_matrix

    def _solve_weights(self, gp, design_matrix):
        """ """
        X = design_matrix
        A = np.dot(X.T, self.gp.apply_inverse(X))
        B = np.dot(X.T, self.gp.apply_inverse(self.raw_lc.flux[:, None])[:, 0])
        w = np.linalg.solve(A, B)
        m = np.dot(X, w)

        return m

    def _neg_log_like(self, params, design_matrix):
        """ """
        self.gp.set_parameter_vector(params)
        m = self._solve_weights(self.gp, design_matrix)
        return -self.gp.log_likelihood(self.raw_lc.flux - m)

    def _grad_neg_log_like(self, params, design_matrix):
        """ """
        self.gp.set_parameter_vector(params)
        m = self._solve_weights(self.gp, design_matrix)
        return -self.gp.grad_log_likelihood(self.raw_lc.flux - m)[1]

    def optimize(self, design_matrix, method="L-BFGS-B"):
        """ """
        self.optimized = True
        solution = minimize(self._neg_log_like, self.gp.get_parameter_vector(),
                            method=method, bounds=self.gp.get_parameter_bounds(),
                            jac=self._grad_neg_log_like, args=(design_matrix))
        self.gp.set_parameter_vector(solution.x)
        return solution

    def create_gp_model(self, cadence_mask=[]):
        """ """
        gpc = GPCorrector(self.raw_lc, kernel="matern32")
        return gpc.gp

    def correct(self, aperture_mask=None, remove_gp_trend=False, **kwargs):
        """ """
        # Create final optimized model
        X = self.create_design_matrix()
        self.gp = self.create_gp_model()
        self.optimize(X)
        m = self._solve_weights(self.gp, X)

        # Store in `~lightkurve.LightCurve` object
        self.corrected_lc = self.raw_lc.copy()
        self.corrected_lc.flux -= m
        self.corrected_lc.flux += np.nanmean(m)

        # Optionally remove long term trend fit by GP
        if remove_gp_trend:
            mu = self.gp.predict(self.corrected_lc.flux, self.corrected_lc.time,
                                 return_cov=False, return_var=False)
            self.corrected_lc.flux -= (mu - np.nanmean(mu))

        return self.corrected_lc

    def diagnose(self, ax=None):
        """ """
        if not self.optimized:
            log.warning("You need to call the `correct` method before diagnosing.")
            return None

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()

        self.raw_lc.scatter(ax=ax, c='r', label='{} (Raw Light Curve)'.format(self.raw_lc.label))
        self.corrected_lc.scatter(ax=ax, c='k', label='{} (PLD-Corrected)'.format(self.raw_lc.label))

        return ax
