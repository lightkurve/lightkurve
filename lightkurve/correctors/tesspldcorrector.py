"""Defines a `TessPLDCorrector` class which provides a simple way to correct a
light curve by utilizing the pixel time series data contained within the
target's own Target Pixel File. Specifically, this corrector is intended to be
used on TESS target pixel files provided by TESSCut.

`TessPLDCorrector` builds upon `RegressionCorrector` by correlating the light curve
against a design matrix composed of the following elements:
* A background light curve to capture the dominant scattered light systematics.
* Background-corrected pixel time series to capture any residual systematics.
* Splines to capture the target's intrinsic variability.
"""
import matplotlib.pyplot as plt
import numpy as np

from . import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix
from .. import MPLSTYLE


__all__ = ['TessPLDCorrector']


class TessPLDCorrector(RegressionCorrector):
    """Correct TESS light curves by detrending against local pixel time series.

    Special case of `.RegressionCorrector` where the `.DesignMatrix` is
    composed of background-corrected pixel time series.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Examples
    --------
    >>> corrector = TessPLDCorrector(tpf)  # doctest: +SKIP
    >>> lc = corrector.correct()  # doctest: +SKIP

    Parameters
    ----------
    tpf : `.TargetPixelFile`
        The target pixel from which a light curve and background model
        will be extracted.
    """
    def __init__(self, tpf, aperture_mask=None):
        if aperture_mask is None:
            aperture_mask = tpf.create_threshold_mask(2)
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        self.tpf = tpf
        self.aperture_mask = aperture_mask
        super(TessPLDCorrector, self).__init__(lc=lc)

    def __repr__(self):
        return 'TessPLDCorrector (LC: {})'.format(self.lc.label)

    def _create_design_matrix(self, background_mask, pixel_components,
                              spline_n_knots, spline_degree):
        """Returns a `DesignMatrixCollection`."""
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

        dm_pixels = DesignMatrix(pixels, name='pixel_series').pca(pixel_components)
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')
        dm_spline = create_spline_matrix(self.lc.time,
                                         n_knots=spline_n_knots,
                                         degree=spline_degree).append_constant()
        dm = DesignMatrixCollection([dm_pixels, dm_bkg, dm_spline])
        return dm

    def correct(self, pixel_components=3, spline_n_knots=100, spline_degree=3,
                background_mask=None, restore_trend=True, **kwargs):
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
                                        spline_degree=spline_degree)
        clc = super(TessPLDCorrector, self).correct(dm, **kwargs)
        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
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
            self.corrected_lc[~self.cadence_mask].scatter(
                                            normalize=False, c='r', marker='x',
                                            s=10, label='Outliers', ax=ax)
            self.corrected_lc.plot(normalize=False, label='Corrected', ax=ax, c='k')
        return axs
