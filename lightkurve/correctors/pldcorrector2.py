"""Defines PLDCorrector
"""
from __future__ import division, print_function

import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np

from .corrector import Corrector
from .. import MPLSTYLE
from ..utils import LightkurveWarning, suppress_stdout

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']

class PLDCorrector(Corrector):

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

    def _create_gp_model(self, cadence_mask):
        pass

    def _find_weights(self):
        pass

    def correct(self, aperture_mask=None, **kwargs):
        pass

    def optimize(self):
        pass
