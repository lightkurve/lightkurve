"""Defines a `PLDCorrector` class which provides a simple way to correct a
light curve by utilizing the pixel time series data contained within the
target's own Target Pixel File.

`PLDCorrector` builds upon `RegressionCorrector` by correlating the light curve
against a design matrix composed of the following elements:
* A background light curve to capture the dominant scattered light systematics.
* Background-corrected pixel time series to capture any residual systematics.
* Splines to capture the target's intrinsic variability.
"""
import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.utils.decorators import deprecated, deprecated_renamed_argument

from .designmatrix import DesignMatrix, DesignMatrixCollection, \
                          SparseDesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix, create_sparse_spline_matrix
from .. import MPLSTYLE
from ..targetpixelfile import KeplerTargetPixelFile
from ..utils import LightkurveDeprecationWarning


log = logging.getLogger(__name__)


__all__ = ['PLDCorrector', 'TessPLDCorrector']


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

    def create_design_matrix(self, pld_order=3, pca_components=16,
                             background_aperture_mask='background', pld_aperture_mask=None,
                             spline_n_knots=100, spline_degree=3, sparse=False):
        """Returns a `.DesignMatrixCollection` containing a `DesignMatrix` object
        for the background regressors, the PLD pixel component regressors, and
        the spline regressors.

        If the parameters `pld_order` and `pca_components` are None, their
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
        pca_components : int or tuple of int
            Number of terms added to the design matrix for each order of PLD
            pixel fluxes. Increasing this value may provide higher precision
            at the expense of slower speed and/or overfitting.
            If performing PLD with `pld_order > 1`, `pca_components` can be
            a tuple containing the number of terms for each order of PLD.
            If a single int is passed, the same number of terms will be used
            for each order. If zero is passed, PCA will not be performed.
            Defaults to 16 for K2 and 8 for TESS.
        background_aperture_mask : array-like or None
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
        sparse : bool
            Whether to create `SparseDesignMatrix`.

        Returns
        -------
        dm : `.DesignMatrixCollection`
            `.DesignMatrixCollection` containing pixel, background, and spline
            components.
        """
        # Validate the inputs
        pld_aperture_mask = self.tpf._parse_aperture_mask(pld_aperture_mask)
        background_aperture_mask = self.tpf._parse_aperture_mask(background_aperture_mask)
        self.pld_aperture_mask = pld_aperture_mask
        self.background_aperture_mask = background_aperture_mask

        if sparse:
            DMC = SparseDesignMatrixCollection
            spline = create_sparse_spline_matrix
        else:
            DMC = DesignMatrixCollection
            spline = create_spline_matrix

        # First, we estimate the per-pixel background flux over time
        bkg = self.tpf.estimate_background(aperture_mask=background_aperture_mask)
        self.background_estimate = bkg

        # Background-subtracted, flux-normalized pixel time series
        regressors = self.tpf.flux[:, pld_aperture_mask].reshape(len(self.tpf.flux), -1)
        regressors = regressors - bkg.flux.reshape(-1,1) * pld_aperture_mask.sum() * u.pixel
        regressors = np.array([r[np.isfinite(r)] for r in regressors])
        regressors = np.array([r / f for r,f in zip(regressors, self.lc.flux.value)])

        # Use the DesignMatrix infrastructure to apply PCA to the regressors.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*low rank.*')
            regressors_dm = DesignMatrix(regressors)
        if isinstance(pca_components, (tuple, list)):
            ncomp = pca_components[0]
        else:
            ncomp = pca_components
        if ncomp > 0:
            regressors_dm = regressors_dm.pca(ncomp)
        regressors_pld = regressors_dm.values


        # Create a DesignMatrix for each PLD order
        all_pld = []
        for order in range(1, pld_order+1):
            reg_n = np.product(list(multichoose(regressors_pld.T, order)), axis=1).T
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*low rank.*')
                pld_n = DesignMatrix(reg_n)
            # Apply PCA. Check if pca_components has an entry for each order,
            # otherwise use pca_components for PCA of higher order matrices.
            if isinstance(pca_components, (tuple, list)):
                ncomp = pca_components[order-1]
            else:
                ncomp = pca_components
            if ncomp > 0:
                pld_n = pld_n.pca(ncomp)
            all_pld.append(pld_n)

        # Create the collection of DesignMatrix objects.
        # DesignMatrix 1 contains the PLD pixel series
        dm_pixels = DesignMatrixCollection(all_pld).to_designmatrix(name='pixel_series')
        # DesignMatrix 2 contains the average per-pixel background flux
        # The prior on the background flux is set equal to the number of pixels
        # in the light curve aperture mask; this assumes the background is additive.
        bkg_prior_mu = self.tpf._parse_aperture_mask(self.lc.meta['aperture_mask']).sum()
        dm_bkg = DesignMatrix(bkg.flux.value, name='background_model',
                              prior_mu=bkg_prior_mu, prior_sigma=1)
        # DesignMatrix 3 contains splines plus a constant
        dm_spline = spline(self.lc.time.value,
                           n_knots=spline_n_knots,
                           degree=spline_degree).append_constant()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Not all matrices are `SparseDesignMatrix` objects..*')
            dm_collection = DMC([dm_pixels, dm_bkg, dm_spline])
        return dm_collection

    @deprecated_renamed_argument('n_pca_terms', 'pca_components', '2.0', warning_type=LightkurveDeprecationWarning)
    @deprecated_renamed_argument('use_gp', None, '2.0', warning_type=LightkurveDeprecationWarning)
    @deprecated_renamed_argument('gp_timescale', None, '2.0', warning_type=LightkurveDeprecationWarning)
    @deprecated_renamed_argument('aperture_mask', None, '2.0', warning_type=LightkurveDeprecationWarning)
    def correct(self, pld_order=None, pca_components=None,
                background_aperture_mask='background', pld_aperture_mask=None,
                spline_n_knots=40, spline_degree=5, restore_trend=True,
                sparse=False, cadence_mask=None, sigma=5, niters=5,
                propagate_errors=False, use_gp=None, gp_timescale=None,
                aperture_mask=None):
        """Returns a systematics-corrected light curve.

        If the parameters `pld_order` and `pca_components` are None, their
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
        pca_components : int or tuple of int
            Number of terms added to the design matrix for each order of PLD
            pixel fluxes. Increasing this value may provide higher precision
            at the expense of slower speed and/or overfitting.
            If performing PLD with `pld_order > 1`, `pca_components` can be
            a tuple containing the number of terms for each order of PLD.
            If a single int is passed, the same number of terms will be used
            for each order.  If zero is passed, PCA will not be performed.
        background_aperture_mask : array-like or None
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
        restore_trend : bool
            Whether to restore the long term spline trend to the light curve.
        sparse : bool
            Whether to create `SparseDesignMatrix`.
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
        use_gp, gp_timescale : DEPRECATED
            As of Lightkurve v2.0 PLDCorrector uses splines instead of Gaussian Processes.
        aperture_mask : DEPRECATED
            As of Lightkurve v2.0 the `aperture_mask` parameter needs to be
            passed to the class constructor.

        Returns
        -------
        clc : `.LightCurve`
            Noise-corrected `.LightCurve`.
        """
        self.restore_trend = restore_trend

        # Set mission-specific values for pld_order and pca_components
        if pld_order is None:
            if isinstance(self.tpf, KeplerTargetPixelFile):
                pld_order = 3
            else:
                pld_order = 1
        if pca_components is None:
            if isinstance(self.tpf, KeplerTargetPixelFile):
                pca_components = 16
            else:
                pca_components = 7

        dm = self.create_design_matrix(background_aperture_mask=background_aperture_mask,
                                       pld_aperture_mask=pld_aperture_mask,
                                       pld_order=pld_order,
                                       pca_components=pca_components,
                                       spline_n_knots=spline_n_knots,
                                       spline_degree=spline_degree,
                                       sparse=sparse)

        clc = super().correct(dm, cadence_mask=cadence_mask, sigma=sigma,
                              niters=niters, propagate_errors=propagate_errors)
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
        if not getattr(self, 'corrected_lc'):
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
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label='original')
            clc[self.outlier_mask].scatter(normalize=False, c='r', marker='x',
                                            s=10, label='outlier_mask', ax=ax)
            clc[~self.cadence_mask].scatter(normalize=False, c='dodgerblue',
                                            marker='x', s=10, label='~cadence_mask',
                                            ax=ax)
            clc.plot(normalize=False, label='corrected', ax=ax, c='k')
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
                          title='aperture_mask')
            # Show background mask
            ax = axs[1]
            self.tpf.plot(ax=ax, show_colorbar=False,
                          aperture_mask=self.background_aperture_mask,
                          title='background_aperture_mask')
            # Show PLD pixel mask
            ax = axs[2]
            self.tpf.plot(ax=ax, show_colorbar=False,
                          aperture_mask=self.pld_aperture_mask,
                          title='pld_aperture_mask')
        return axs


# `TessPLDCorrector` was briefly introduced in Lightkurve v1.9
# but was removed in v2.0 in favor of a single generic `PLDCorrector`.
@deprecated("2.0", alternative="PLDCorrector",
            warning_type=LightkurveDeprecationWarning)
class TessPLDCorrector(PLDCorrector):
    pass
