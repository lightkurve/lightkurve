"""Defines PyMCPLDCorrector

TODO
----
* Add input validation on pld_order etc.
* Remove saturated pixels from design matrix.
* Add convenience method to plot the design matrix.
"""
from __future__ import division, print_function

import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np

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
        """Returns normalized pixel flux values in the PLD mask re-arranged
        into a 2D matrix with shape (n_cadences, n_pixels_in_pld_mask).
        
        This matrix will form the basis of the PLD regressor design matrix
        and is often called the first order component.

        The matrix returned is guaranteed to be free of NaN values.

        Returns
        -------
        matrix : numpy array
            First order PLD design matrix.
        """ 
        # Re-arrange the cube of flux values observed in a user-specified mask
        # into a 2D matrix of shape (n_cadences, n_pixels_in_mask)
        matrix = self.tpf.flux[:, self.pld_aperture_mask]
        assert matrix.shape == (len(self.raw_lc.time), self.pld_aperture_mask.sum())
        # Remove all NaN or Inf values
        matrix = matrix[:, np.isfinite(matrix).all(axis=0)]
        # Normalize each cadence to 1 by dividing by the per-cadence pixel sums
        matrix = matrix / np.sum(matrix, axis=-1)[:, None]
        return matrix

    def create_design_matrix(self, pld_order=2, n_pca_terms=10, include_column_of_ones=False):
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

        TODO
        ----
        * The design matrix can be improved by rejecting pixels which are
          saturated, and optionally including the collapsed sums of their CCD
          columns instead.
        * It is not clear whether the inclusion of a column vector of ones
          is necessary for numerical stability.

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

    def plot_design_matrix(self):
        pass

    def get_pymc_model(self):
        pass
    
    def optimize(self):
        pass

    def sample(self):
        pass
