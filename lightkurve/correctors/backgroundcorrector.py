"""Defines a BackgroundCorrector class which provides a simple way to correct a
light curve by utilizing the pixel time series data contained within the
target's own Target Pixel File. Specifically, this corrector is intended to be
used on TESS target pixel files provided by TESSCut.

PixelCorrector builds upon RegressionCorrector by correlating the light curve
against a design matrix composed of the following elements:
* A background light curve to capture the dominant scattered light systematics.
* Background-corrected pixel time series to capture any residual systematics.
* Splines to capture the target's intrinsic variability.
"""
import numpy as np

from . import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix


__all__ = ['BackgroundCorrector']


class BackgroundCorrector(RegressionCorrector):
    """Correct a light curve using local pixel time series.
    
    Special case of `.RegressionCorrector` where the `.DesignMatrix` is
    composed of background-corrected pixel time series.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Examples
    --------
    >>> corrector = BackgroundCorrector(tpf)  # doctest: +SKIP
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
        super(BackgroundCorrector, self).__init__(lc=lc)

    def __repr__(self):
        return 'BackgroundCorrector (LC: {})'.format(self.lc.label)

    def _create_design_matrix(self, background_mask, pixel_components,
                              spline_n_knots, spline_degree):
        """Returns a DesignMatrixCollection."""
        # Estimate the median background over time
        # Subtract the mean image from each cadence
        simple_bkg = (self.tpf.flux - np.nanmean(self.tpf.flux, axis=0))
        # Compute the median background pixel value over time
        simple_bkg = np.nanmedian(simple_bkg[:, background_mask], axis=1)
        simple_bkg -= np.percentile(simple_bkg, 5)

        # Background-corrected pixel time series
        pixels = (self.tpf.flux.transpose([1, 2, 0]) - simple_bkg
                    ).transpose([2, 0, 1])[:, background_mask]

        dm_pixels = DesignMatrix(pixels, name='pixel_series').pca(pixel_components)
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')
        dm_spline = create_spline_matrix(self.lc.time, n_knots=spline_n_knots, degree=spline_degree).append_constant()
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

        dm = self._create_design_matrix(background_mask=background_mask,
                                        pixel_components=pixel_components,
                                        spline_n_knots=spline_n_knots,
                                        spline_degree=spline_degree)
        clc = super(BackgroundCorrector, self).correct(dm, **kwargs)
        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
        return clc
