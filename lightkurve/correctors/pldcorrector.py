"""Defines PLDCorrector
"""
import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np
import matplotlib.pyplot as plt

from .designmatrix import DesignMatrix, DesignMatrixCollection, \
                          SparseDesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix, create_sparse_spline_matrix
from .. import MPLSTYLE
from ..targetpixelfile import KeplerTargetPixelFile

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']


class PLDCorrector(RegressionCorrector):
    r"""Implements the Pixel Level Decorrelation (PLD) systematics removal method.

    Special case of `.RegressionCorrector` where the `.DesignMatrix` is
    composed of background-corrected pixel time series.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

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
    To solve for the PLD model, we need to minimize the difference squared

    .. math::
        \chi^2 = \sum_i \frac{(y_i - m_i)^2}{\sigma_i^2},

    where :math:`y_i` is the observed flux value at time :math:`t_i`, by solving

    .. math::
        \frac{\partial \chi^2}{\partial a_l} = 0.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Examples
    --------
    Download the pixel data for GJ 9827 and obtain a PLD-corrected light curve:

    >>> import lightkurve as lk
    >>> tpf = lk.search_targetpixelfile("GJ9827").download() # doctest: +SKIP
    >>> corrector = tpf.to_corrector('pld') # doctest: +SKIP
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
    def __init__(self, tpf, aperture_mask=None):
        self.tpf = tpf
        if aperture_mask is None:
            aperture_mask = tpf.create_threshold_mask(3)
        self.aperture_mask = aperture_mask
        lc = self.tpf.to_lightcurve(aperture_mask=aperture_mask)
        super().__init__(lc=lc)

    def __repr__(self):
        return 'PLDCorrector (ID: {})'.format(self.lc.label)

    def create_design_matrix(self, pld_order=3, pixel_components=16,
                             background_mask='background', pld_aperture_mask=None,
                             spline_n_knots=100, spline_degree=3,
                             n_pca_terms=6, sparse=False):
        """Returns a `.DesignMatrixCollection` containing a `DesignMatrix` object
        for the background regressors, the PLD pixel component regressors, and
        the spline regressors.

        If the parameters `pld_order` and `pixel_components` are None, their
        value will be assigned based on the mission. K2 and TESS experience
        different dominant sources of noise, and require different defaults.
        For information about how the defaults were chosen, see Pull Request #746.

        Parameters
        ----------
        pld_order : int
            The order of Pixel Level De-correlation to be performed. First order
            (`n=1`) uses only the pixel fluxes to construct the design matrix.
            Higher order populates the design matrix with columns constructed
            from the products of pixel fluxes.
        pixel_components : int
            Number of principal components derived from the background pixel
            time series to utilize.
        background_mask : array-like or None
            A boolean array flagging the background pixels such that `True` means
            that the pixel will be used to generate the background systematics model.
            If `None`, all pixels which are fainter than 1-sigma above the median
            flux will be used.
        pld_aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used when selecting the PLD basis vectors.
            If `None` or `all` are passed in, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
        spline_n_knots : int
            Number of knots in spline.
        spline_degree : int
            Polynomial degree of spline.
        n_pca_terms : int
            Number of terms added to the design matrix from each order of PLD
            when performing Principal Component Analysis for models higher than
            first order. Increasing this value may provide higher precision at
            the expense of computational time.
        sparse : bool
            Whether to create `SparseDesignMatrix`.

        Returns
        -------
        dm : `.DesignMatrixCollection`
            `.DesignMatrixCollection` containing pixel, background, and spline
            components.
        """
        background_mask = self.tpf._parse_aperture_mask(background_mask)
        self.background_mask = background_mask

        DMC, spline = DesignMatrixCollection, create_spline_matrix
        if sparse:
            DMC, spline = SparseDesignMatrixCollection, create_sparse_spline_matrix
        # First, we estimate the per-pixel background flux over time by
        # (i) subtracting a mean image from each cadence;
        # (ii) computing the median pixel value in the residual images;
        # (iii) assume that the 5%-percentile of those medians gives us the
        # exact background level. This assumption appears to work well for TESS
        # but it has not been validated in detail yet.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            simple_bkg = (self.tpf.flux - np.nanmean(self.tpf.flux, axis=0))
        simple_bkg = np.nanmedian(simple_bkg[:, background_mask], axis=1)
        simple_bkg -= np.percentile(simple_bkg, 5)

        # Parse PLD aperture mask
        self.pld_pixel_mask = self.tpf._parse_aperture_mask(pld_aperture_mask)

        # Background-subtracted, flux-normalized pixel time series
        regressors = self.tpf.flux[:, self.pld_pixel_mask].reshape(len(self.tpf.flux), -1)
        regressors = regressors - simple_bkg.reshape(-1,1)
        regressors = np.array([r[np.isfinite(r)] for r in regressors])
        regressors = np.array([r / f for r,f in zip(regressors, self.lc.flux.value)])

        # Create first order design matrix
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pld_1 = DesignMatrix(regressors).pca(pixel_components)

        # Create higher order matrix
        all_pld = [pld_1]
        for i in range(2, pld_order+1):
            # This step creates higher order products of pixel components,
            # from 2nd to nth order
            reg_n = np.product(list(multichoose(pld_1.values.T, i)), axis=1).T
            # Apply PCA before merging into single PLD matrix
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pld_n = DesignMatrix(reg_n).pca(pixel_components)
            all_pld.append(pld_n)

        # Collect each matrix
        dm_pixels = DesignMatrixCollection(all_pld).to_designmatrix(name='pixel_series')
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')
        dm_spline = spline(self.lc.time.value, n_knots=spline_n_knots,
                             degree=spline_degree).append_constant()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dm = DMC([dm_pixels, dm_bkg, dm_spline])

        return dm

    def correct(self, pld_order=None, pixel_components=None,
                background_mask='background', pld_aperture_mask=None, spline_n_knots=40,
                spline_degree=5, n_pca_terms=8, restore_trend=True, sparse=False,
                **kwargs):
        """Returns a systematics-corrected light curve.

        If the parameters `pld_order` and `pixel_components` are None, their
        value will be assigned based on the mission. K2 and TESS experience
        different dominant sources of noise, and require different defaults.
        For information about how the defaults were chosen, see PR #746 at
        https://github.com/KeplerGO/lightkurve/pull/746#issuecomment-658458270

        Parameters
        ----------
        pld_order : int
            The order of Pixel Level De-correlation to be performed. First order
            (`n=1`) uses only the pixel fluxes to construct the design matrix.
            Higher order populates the design matrix with columns constructed
            from the products of pixel fluxes. Default 3 for K2 and 1 for TESS.
        pixel_components : int
            Number of principal components derived from the background pixel
            time series to utilize. Default 16 for K2 and 8 for TESS.
        background_mask : array-like or None
            A boolean array flagging the background pixels such that `True` means
            that the pixel will be used to generate the background systematics model.
            If `None`, all pixels which are fainter than 1-sigma above the median
            flux will be used.
        pld_aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used when selecting the PLD basis vectors.
            If `None` or `all` are passed in, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
        spline_n_knots : int
            Number of knots in spline.
        spline_degree : int
            Polynomial degree of spline.
        n_pca_terms : int
            Number of terms added to the design matrix from each order of PLD
            when performing Principal Component Analysis for models higher than
            first order. Increasing this value may provide higher precision at
            the expense of computational time.
        restore_trend : bool
            Whether to restore the long term spline trend to the light curve.
        sparse : bool
            Whether to create `SparseDesignMatrix`.
        **kwargs : dict
            Extra parameters to be passed to `RegressionCorrector.correct`.

        Returns
        -------
        clc : `.LightCurve`
            Noise-corrected `.LightCurve`.
        """
        self.restore_trend = restore_trend

        # Set mission-specific values for pld_order and pixel_components
        if pld_order is None:
            if isinstance(self.tpf, KeplerTargetPixelFile):
                pld_order = 3
            else:
                pld_order = 1
        if pixel_components is None:
            if isinstance(self.tpf, KeplerTargetPixelFile):
                pixel_components = 16
            else:
                pixel_components = 7

        dm = self.create_design_matrix(background_mask=background_mask,
                                       pld_aperture_mask=pld_aperture_mask,
                                       pld_order=pld_order,
                                       n_pca_terms=n_pca_terms,
                                       pixel_components=pixel_components,
                                       spline_n_knots=spline_n_knots,
                                       spline_degree=spline_degree,
                                       sparse=sparse)

        clc = super().correct(dm, **kwargs)
        if restore_trend:
            clc += (self.diagnostic_lightcurves['spline']
                    - np.median(self.diagnostic_lightcurves['spline'].flux))
        return clc

    def diagnose(self):
        """Returns diagnostic plots to assess the most recent call to `correct()`.
        If `correct()` has not yet been called, a ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if not hasattr(self, 'corrected_lc'):
            raise ValueError('You need to call the `correct()` method '
                             'before you can call `diagnose()`.')
        names = self.diagnostic_lightcurves.keys()

        # Plot the right version of corrected light curve
        if self.restore_trend:
            clc = self.corrected_lc \
                  + self.diagnostic_lightcurves['spline'] \
                  - np.median(self.diagnostic_lightcurves['spline'].flux)
        else:
            clc = self.corrected_lc

        # Use lightkurve plotting style
        with plt.style.context(MPLSTYLE):
            # Plot background model
            _, axs = plt.subplots(3, figsize=(10, 9), sharex=True)
            ax = axs[0]
            self.lc.plot(ax=ax, normalize=False, label='original', alpha=0.4)
            for key in ['background_model']:
                tmplc = self.diagnostic_lightcurves[key] \
                        - np.median(self.diagnostic_lightcurves[key].flux) \
                        + np.median(self.lc.flux)
                tmplc.plot(ax=ax)
            ax.set_xlabel('')

            # Plot pixel and spline components
            ax = axs[1]
            clc.plot(ax=ax, normalize=False, label='corrected', alpha=0.4)
            for key in names:
                if key in ['pixel_series', 'spline']:
                    tmplc = self.diagnostic_lightcurves[key] \
                            - np.median(self.diagnostic_lightcurves[key].flux) \
                            + np.median(self.lc.flux)
                    tmplc.plot(ax=ax)
            ax.set_xlabel('')

            # Plot final corrected light curve with outliers marked
            ax = axs[2]
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label='Original')
            clc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x',
                                            s=10, label='Outliers', ax=ax)
            clc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return axs

    def diagnose_masks(self):
        """Show different aperture masks used by PLD in the most recent call to
        `correct()`. If `correct()` has not yet been called, a ``ValueError``
        will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if not hasattr(self, 'corrected_lc'):
            raise ValueError('You need to call the `correct()` method '
                             'before you can call `diagnose()`.')

        # Use lightkurve plotting style
        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            # Show light curve aperture mask
            ax = axs[0]
            self.tpf.plot(ax=ax, show_colorbar=False,
                          aperture_mask=self.aperture_mask,
                          title='Light Curve Mask')
            # Show background mask
            ax = axs[1]
            self.tpf.plot(ax=ax, show_colorbar=False,
                          aperture_mask=self.background_mask,
                          title='Background Mask')
            # Show PLD pixel mask
            ax = axs[2]
            self.tpf.plot(ax=ax, show_colorbar=False,
                          aperture_mask=self.pld_pixel_mask,
                          title='PLD Mask')
        return axs
