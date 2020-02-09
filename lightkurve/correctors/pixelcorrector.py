import numpy as np

from . import DesignMatrix, DesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .sffcorrector import _get_spline_dm


__all__ = ['PixelCorrector']


class PixelCorrector(RegressionCorrector):
    """Special case of `.RegressionCorrector` where the `.DesignMatrix` is
    composed of background-corrected pixel time series.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Parameters
    ----------
    lc : `.LightCurve`
        The light curve that needs to be corrected.
    """
    def __init__(self, tpf, aperture_mask=None, background_mask=None):
        if aperture_mask is None:
            aperture_mask = tpf.create_threshold_mask(2)
        if background_mask is None:
            # Default to pixels not 1-sigma above the background
            background_mask = ~tpf.create_threshold_mask(1, reference_pixel=None)

        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)

        self.tpf = tpf
        self.aperture_mask = aperture_mask
        self.background_mask = background_mask
        super(PixelCorrector, self).__init__(lc=lc)

    def __repr__(self):
        return 'PixelCorrector (LC: {})'.format(self.lc.label)

    def correct(self, restore_trend=True, **kwargs):
        """Find the best fit correction for the light curve.
        """
        # Estimate the median background over time
        # Subtract the mean image from each cadence
        simple_bkg = (self.tpf.flux - np.nanmean(self.tpf.flux, axis=0))
        # Compute the median background pixel value over time
        simple_bkg = np.nanmedian(simple_bkg[:, self.background_mask], axis=1)
        simple_bkg -= np.percentile(simple_bkg, 5)

        # Background-corrected pixel time series
        pixels = (self.tpf.flux.transpose([1, 2, 0]) - simple_bkg).transpose([2, 0, 1])[:, self.background_mask]

        dm_pixels = DesignMatrix(pixels, name='pixel_series').pca(3)
        dm_bkg = DesignMatrix(simple_bkg, name='background_model')
        dm_spline = _get_spline_dm(self.lc.time, 100).append_constant()

        dm = DesignMatrixCollection([dm_pixels, dm_bkg, dm_spline])

        clc = super(PixelCorrector, self).correct(dm, **kwargs)

        if restore_trend:
            clc += self.diagnostic_lightcurves['spline']
        return clc
