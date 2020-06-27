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
        self.nanmask &= np.isfinite(rawflux.value)
        self.nanmask &= np.isfinite(rawflux_err.value)
        self.nanmask &= np.abs(rawflux_err.value) > 1e-12

        # apply nan mask
        self.flux = flux[self.nanmask]
        self.flux_err = flux_err[self.nanmask]
        self.time = time[self.nanmask]
        self.rawflux = rawflux[self.nanmask]
        self.rawflux_err = rawflux_err[self.nanmask]
        self.lc = lc[self.nanmask]

        super(PLDCorrector, self).__init__(lc=self.lc)

    def __repr__(self):
        return 'PLDCorrector (LC: {})'.format(self.lc.label)

    @property
    def X(self):
        return self.dm

    def create_design_matrix(self, background_mask=None, pld_order=1, n_pca_terms=6,
                             pixel_components=3, spline_n_knots=100, spline_degree=3, sparse=False):
        """Returns a `DesignMatrixCollection`."""

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

        # Flux-normalzied pixel time series
        regressors = self.tpf.flux.reshape(len(self.tpf.flux), -1)
        regressors = np.array([r[np.isfinite(r)] for r in regressors])
        regressors = np.array([r / f for r,f in zip(regressors, self.lc.flux.value)])

        # Create first order design matrix
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pld_1 = DesignMatrix(regressors).pca(pixel_components)

        # Create higher order matrix
        all_pld = [pld_1]
        for i in range(2, pld_order+1):
            reg_n = np.product(list(multichoose(pld_1.values.T, i)), axis=1).T
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pld_n = DesignMatrix(reg_n).pca(pixel_components)
            all_pld.append(pld_n)

        dm_pixels = DesignMatrixCollection(all_pld).to_designmatrix(name='pixel_series')
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')

        dm_spline = spline(self.lc.time.value, n_knots=spline_n_knots,
                             degree=spline_degree).append_constant()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dm = DMC([dm_pixels, dm_bkg, dm_spline])

        self.dm = dm
        return dm

    def correct(self, dm=None, pld_order=1, background_mask=None, pixel_components=3,
                spline_n_knots=100, spline_degree=3,
                n_pca_terms=10, restore_trend=True, sparse=False, **kwargs):
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

        if dm is None:
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
        self.dm = dm
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
            raise ValueError('Please call the `correct()` method before trying to diagnose.')

        names = self.diagnostic_lightcurves.keys()

        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(3, figsize=(10, 9), sharex=True)
            ax = axs[0]
            self.lc.plot(ax=ax, normalize=False, label='original', alpha=0.4)
            for key in ['background_model']:
                (self.diagnostic_lightcurves[key] - np.median(self.diagnostic_lightcurves[key].flux) + np.median(self.lc.flux)).plot(ax=ax)
            ax.set_xlabel('')

            ax = axs[1]
            self.corrected_lc.plot(ax=ax, normalize=False, label='corrected', alpha=0.4)
            for key in names:
                if key in ['pixel_series', 'spline']:
                    (self.diagnostic_lightcurves[key] - np.median(self.diagnostic_lightcurves[key].flux) + np.median(self.lc.flux)).plot(ax=ax)
            ax.set_xlabel('')

            ax = axs[2]
            self.lc.plot(ax=ax, normalize=False, alpha=0.2, label='Original')
            self.corrected_lc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x',
                                      s=10, label='Outliers', ax=ax)
            self.corrected_lc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return axs


class TessPLDCorrector(PLDCorrector):
    """Correct TESS light curves by detrending against local pixel time series.

    Subclass of `.PLDCorrector`, a version of the `.RegressionCorrector` class
    in which the `.DesignMatrix` is constructed from pixel time series.

    Parameters
    ----------
    tpf : `.TargetPixelFile`
        The target pixel from which a light curve and background model
        will be extracted.
    """

    def __init__(self, tpf):
        super(TessPLDCorrector, self).__init__(tpf)

    def correct(self, dm=None, pld_order=1, background_mask=None, pixel_components=3,
                spline_n_knots=100, spline_degree=3,
                n_pca_terms=10, restore_trend=True, sparse=False, **kwargs):
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

        clc = super(TessPLDCorrector, self).correct(dm=dm,
                                                    pld_order=pld_order,
                                                    background_mask=background_mask,
                                                    pixel_components=pixel_components,
                                                    spline_n_knots=spline_n_knots,
                                                    spline_degree=spline_degree,
                                                    n_pca_terms=n_pca_terms,
                                                    restore_trend=restore_trend,
                                                    sparse=sparse,
                                                    **kwargs)

        return clc


class KeplerPLDCorrector(PLDCorrector):
    """Correct Kepler light curves by detrending against local pixel time series.

    Subclass of `.PLDCorrector`, a version of the `.RegressionCorrector` class
    in which the `.DesignMatrix` is constructed from pixel time series.

    Parameters
    ----------
    tpf : `.TargetPixelFile`
        The target pixel from which a light curve and background model
        will be extracted.
    """

    def __init__(self, tpf):
        super(KeplerPLDCorrector, self).__init__(tpf)

    def correct(self, dm=None, pld_order=2, background_mask=None, pixel_components=15,
                spline_n_knots=100, spline_degree=3,
                n_pca_terms=10, restore_trend=True, sparse=False, **kwargs):
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

        clc = super(KeplerPLDCorrector, self).correct(dm=dm,
                                                      pld_order=pld_order,
                                                      background_mask=background_mask,
                                                      pixel_components=pixel_components,
                                                      spline_n_knots=spline_n_knots,
                                                      spline_degree=spline_degree,
                                                      n_pca_terms=n_pca_terms,
                                                      restore_trend=restore_trend,
                                                      sparse=sparse,
                                                      **kwargs)

        return clc
