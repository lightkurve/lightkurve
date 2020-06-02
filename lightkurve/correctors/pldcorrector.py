"""Defines PLDCorrector
"""
from __future__ import division, print_function

import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np
import matplotlib.pyplot as plt

from .corrector import Corrector
from .designmatrix import DesignMatrix, DesignMatrixCollection, SparseDesignMatrix, SparseDesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix, create_sparse_spline_matrix
from .. import MPLSTYLE

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector', 'TessPLDCorrector', 'KeplerPLDCorrector']


class PLDCorrector(RegressionCorrector):
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
    def __init__(self, tpf, aperture_mask=None):
        self.tpf = tpf
        if aperture_mask is None:
            aperture_mask = tpf.create_threshold_mask(2)

        lc = self.tpf.to_lightcurve(aperture_mask=aperture_mask)

        flux = tpf.flux
        flux_err = tpf.flux_err
        time = tpf.time
        rawflux = lc.flux
        rawflux_err = lc.flux_err

        # create nan mask
        self.nanmask = np.isfinite(time)
        self.nanmask &= np.isfinite(rawflux)
        self.nanmask &= np.isfinite(rawflux_err)
        self.nanmask &= np.abs(rawflux_err) > 1e-12

        # apply nan mask
        self.flux = flux[self.nanmask]
        self.flux_err = flux_err[self.nanmask]
        self.time = time[self.nanmask]
        self.rawflux = rawflux[self.nanmask]
        self.rawflux_err = rawflux_err[self.nanmask]
        self.lc = lc[self.nanmask]

        self.rawflux = self.lc.flux
        self.rawflux_err = self.lc.flux_err

        # create nan mask
        self.nanmask = np.isfinite(self.time)
        self.nanmask &= np.isfinite(self.rawflux)
        self.nanmask &= np.isfinite(self.rawflux_err)
        self.nanmask &= np.abs(self.rawflux_err) > 1e-12

        super(PLDCorrector, self).__init__(lc=self.lc)

    def __repr__(self):
        return 'PLDCorrector (LC: {})'.format(self.lc.label)

    @property
    def X(self):
        return self.dm

<<<<<<< HEAD
    def create_design_matrix(self, background_mask=None, pld_order=1, n_pca_terms=6,
                             pixel_components=3, spline_n_knots=100, spline_degree=3, sparse=False):
        """Returns a `DesignMatrixCollection`."""
=======
        Returns
        -------
        X : `.DesignMatrix`
            Matrix of column vectors to perform linear regression.
        """
        if tpf is None:
            tpf = self.tpf

        # parse apertures
        if aperture_mask is None:
            aperture_mask = tpf._parse_aperture_mask('threshold')
            log.debug('No aperture mask provided; using a threshold mask.')
        else:
            aperture_mask = tpf._parse_aperture_mask(aperture_mask)

        if pld_aperture_mask is None:
            pld_aperture_mask = ~tpf._parse_aperture_mask('threshold')
            log.debug('No PLD aperture mask provided; using a threshold mask.')
        else:
            pld_aperture_mask = tpf._parse_aperture_mask(pld_aperture_mask)

        # generate flux light curve from desired pixels
        lc = self.lc

        # find pixel bounds of aperture on tpf
        xmin, xmax = min(np.where(pld_aperture_mask)[0]),  max(np.where(pld_aperture_mask)[0])
        ymin, ymax = min(np.where(pld_aperture_mask)[1]),  max(np.where(pld_aperture_mask)[1])

        # crop data cube to include only desired pixels
        # this is required for superstamps to ensure matrix is invertable
        cropped_flux = self.flux[:, xmin:xmax+1, ymin:ymax+1]
        cropped_flux_err = self.flux_err[:, xmin:xmax+1, ymin:ymax+1]
        cropped_pld_aperture = pld_aperture_mask[xmin:xmax+1, ymin:ymax+1]

        # calculate errors (ignore warnings related to zero or negative errors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = np.nansum(cropped_flux_err[:, cropped_pld_aperture]**2, axis=1)**0.5

        # build initial 1st order PLD design matrix
        regressors = cropped_flux[:, cropped_pld_aperture]
>>>>>>> add option to correct with gp

        if background_mask is None:
            # Default to pixels <1-sigma above the background
            background_mask = ~self.tpf.create_threshold_mask(1, reference_pixel=None)
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
        simple_bkg = (self.tpf.flux - np.nanmean(self.tpf.flux, axis=0))
        simple_bkg = np.nanmedian(simple_bkg[:, background_mask], axis=1)
        simple_bkg -= np.percentile(simple_bkg, 5)

        # Background-corrected pixel time series
        pixels = (self.tpf.flux.transpose([1, 2, 0]) - simple_bkg
                    ).transpose([2, 0, 1])[:, background_mask]

        # make sure no columns have nans
        nanmask = np.isfinite(pixels)
        zipped = zip(pixels, nanmask)
        pixels = np.array([p[n] for p,n in zipped])

        dm_pixels = DesignMatrix(pixels, name='pixel_series').pca(pixel_components)
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')
        dm_spline = spline(self.lc.time, n_knots=spline_n_knots,
                             degree=spline_degree).append_constant()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dm = DMC([dm_pixels.standardize(), dm_bkg.standardize(), dm_spline])

        if pld_order > 1:
            dm = self._create_higher_order_matrix(dm, order=pld_order, n_pca_terms=n_pca_terms)

        self.dm = dm
        return dm

    def correct(self, pld_order=1, pixel_components=3, spline_n_knots=100, spline_degree=3,
                n_pca_terms=10, background_mask=None, restore_trend=True, sparse=False, **kwargs):
        """Returns a systematics-corrected light curve.
        Parameters
        ----------
        pixel_components : int
            Number of principal components derived from the background pixel
            time series to utilize.
        background_mask : array-like or None
            A boolean array flagging the background pixels such that `True` means
            that the pixel will be used to generate the background systematics model.
            If `None`, all pixels which are fainter than 1-sigma above the median
            flux will be used.
        restore_trend : bool
            Whether to restore the long term spline trend to the light curve.
        """
        if background_mask is None:
            # Default to pixels <1-sigma above the background
            background_mask = ~self.tpf.create_threshold_mask(1, reference_pixel=None)
        self.background_mask = background_mask

        dm = self.create_design_matrix(background_mask=background_mask,
                                       pld_order=pld_order,
                                       n_pca_terms=n_pca_terms,
                                       pixel_components=pixel_components,
                                       spline_n_knots=spline_n_knots,
                                       spline_degree=spline_degree,
                                       sparse=sparse)

        clc = super(PLDCorrector, self).correct(dm, **kwargs)
        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
        return clc

    def _correct_with_gp(self, cadence_mask=None, ridge_value=1e-8, **kwargs):
        """ """

        try:
            import celerite
        except ImportError:
            log.error("PLD uses the `celerite` Python package. "
                      "See the installation instructions at "
                      "https://docs.lightkurve.org/about/install.html. "
                      "`use_gp` has been set to `False`.")

        X = self.create_design_matrix(**kwargs)

        # set default transit mask
        if cadence_mask is None:
            cadence_mask = np.ones_like(self.lc.time, dtype=bool)
        M = lambda x: x[cadence_mask[self.nanmask]]

        # mask transits in design matrix
        MX = X.mask(cadence_mask)

        X_gp, gp = MX.apply_gp_inverse(self.lc, ridge_value=ridge_value, return_gp=True,
                                   cadence_mask=cadence_mask[self.nanmask])
        y_gp = gp.apply_inverse(M(self.rawflux)[:, None])[:, 0]

        # compute the coefficients C on the basis vectors;
        # the PLD design matrix will be dotted with C to solve for the noise model.
        A = np.dot(MX.X.T, X_gp)
        B = np.dot(MX.X.T, y_gp)

        A[np.diag_indices_from(A)] += ridge_value
        C = np.linalg.solve(A, B)

        # compute detrended light curve
        model = np.dot(X.X, C)
        self.detrended_flux = self.rawflux - (model - np.nanmean(model))

        # Create and return a new LightCurve object with the corrected flux
        corrected_lc = self.lc.copy()[self.nanmask]
        corrected_lc.flux = self.detrended_flux
        corrected_lc.flux_err = self.flux_err

        return corrected_lc

    def diagnose(self):
        """Returns diagnostic plots to assess the most recent call to `correct()`.
        If `correct()` has not yet been called, a ``ValueError`` will be raised.
        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if not hasattr(self, 'corrected_lc'):
            raise ValueError('Please call the `correct()` method before trying to diagnose.')

        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(3, figsize=(10, 9), sharex=True)
            ax = axs[0]
            self.lc.plot(ax=ax, normalize=False, label='original', alpha=0.4)
            for key in ['background_model']:
                (self.diagnostic_lightcurves[key] - np.median(self.diagnostic_lightcurves[key].flux) + np.median(self.lc.flux)).plot(ax=ax)
            ax.set_xlabel('')

            ax = axs[1]
            self.corrected_lc.plot(ax=ax, normalize=False, label='corrected', alpha=0.4)
            for key in ['pixel_series', 'spline']:
                (self.diagnostic_lightcurves[key] - np.median(self.diagnostic_lightcurves[key].flux) + np.median(self.lc.flux)).plot(ax=ax)
            ax.set_xlabel('')

            ax = axs[2]
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label='Original')
            self.corrected_lc.scatter(normalize=False, c='r', marker='x',
                                      s=10, label='Outliers', ax=ax)
            self.corrected_lc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return axs

    def _create_higher_order_matrix(self, dm, order=2, n_pca_terms=None, prior_mu=0,
                                    prior_sigma=np.inf):
        """Calculate products of columns in `DesignMatrix` and create a new
        `DesignMatrix` for each order of products. Returns a
        `DesignMatrixCollection` with one entry per order.
        Returns
        -------
        `.DesignMatrixCollection`
            Design matrix collection with products of columns appended as new columns.
        """
        # higher order design matrices
        all_dms = [X for X in dm if X.name != 'pixel_series']
        new_dms = [dm['pixel_series']]
        for i in range(2, order+1):
            regressors = np.product(list(multichoose(dm['pixel_series'].values.T, order)), axis=1).T

            # make high order design matrix
            high_order_dm = DesignMatrix(regressors, name=f'PLD Order {i}').standardize()

            # apply PCA
            if n_pca_terms is not None:
                high_order_dm = high_order_dm.pca(n_pca_terms)
            else:
                n_pca_terms = high_order_dm.X.shape[1]

            if isinstance(prior_mu, (int, float)):
                prior_mu = prior_mu * np.ones(n_pca_terms)
            if isinstance(prior_sigma, (int, float)):
                prior_sigma = prior_sigma * np.ones(n_pca_terms)

            high_order_dm.prior_mu = prior_mu
            high_order_dm.prior_sigma = prior_sigma

            new_dms.append(high_order_dm)

        pld_dm = DesignMatrixCollection(new_dms).to_designmatrix(name='pixel_series')
        all_dms.insert(0, pld_dm)

        return DesignMatrixCollection(all_dms)

class TessPLDCorrector(PLDCorrector):
    """
    """

    def __init__(self, tpf):

        super(TessPLDCorrector, self).__init__(tpf)

    def correct(self, pixel_components=3, spline_n_knots=100, spline_degree=3,
                background_mask=None, restore_trend=True, sparse=False, **kwargs):
        """Returns a systematics-corrected light curve.
        Parameters
        ----------
        pixel_components : int
            Number of principal components derived from the background pixel
            time series to utilize.
        background_mask : array-like or None
            A boolean array flagging the background pixels such that `True` means
            that the pixel will be used to generate the background systematics model.
            If `None`, all pixels which are fainter than 1-sigma above the median
            flux will be used.
        restore_trend : bool
            Whether to restore the long term spline trend to the light curve.
        """
        if background_mask is None:
            # Default to pixels <1-sigma above the background
            background_mask = ~self.tpf.create_threshold_mask(1, reference_pixel=None)
        self.background_mask = background_mask

        dm = self._create_design_matrix(background_mask=background_mask,
                                        pixel_components=pixel_components,
                                        spline_n_knots=spline_n_knots,
                                        spline_degree=spline_degree,
                                        sparse=sparse)
        clc = super(TessPLDCorrector, self).correct(dm, **kwargs)
        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
        return clc

class KeplerPLDCorrector(PLDCorrector):
    """
    """

    def __init__(self, tpf):

        super(KeplerPLDCorrector, self).__init__(tpf)

    def correct(self, pld_order=2, pixel_components=15, spline_n_knots=100, spline_degree=3,
                background_mask=None, restore_trend=True, sparse=False, **kwargs):
        """Returns a systematics-corrected light curve.
        Parameters
        ----------
        pixel_components : int
            Number of principal components derived from the background pixel
            time series to utilize.
        background_mask : array-like or None
            A boolean array flagging the background pixels such that `True` means
            that the pixel will be used to generate the background systematics model.
            If `None`, all pixels which are fainter than 1-sigma above the median
            flux will be used.
        restore_trend : bool
            Whether to restore the long term spline trend to the light curve.
        """
        if background_mask is None:
            # Default to pixels <1-sigma above the background
            background_mask = ~self.tpf.create_threshold_mask(1, reference_pixel=None)
        self.background_mask = background_mask

        dm = self.create_design_matrix(background_mask=background_mask,
                                        pixel_components=pixel_components,
                                        spline_n_knots=spline_n_knots,
                                        spline_degree=spline_degree,
                                        sparse=sparse)
        self.dm = dm
        clc = super(TessPLDCorrector, self).correct(dm, **kwargs)
        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
        return clc
=======
    def create_design_matrix(self):
        """ """

        pass
>>>>>>> add option to correct with gp
