"""Defines SFFCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np
from scipy import linalg, interpolate
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip

from .corrector import Corrector

log = logging.getLogger(__name__)

__all__ = ['SFFCorrector']


class SFFCorrector(Corrector):
    """Implements the Self-Flat-Fielding (SFF) systematics removal method.

    This method is described in detail by Vanderburg and Johnson (2014).
    Briefly, the algorithm implemented in this class can be described
    as follows

       (1) Rotate the centroid measurements onto the subspace spanned by the
           eigenvectors of the centroid covariance matrix
       (2) Fit a polynomial to the rotated centroids
       (3) Compute the arclength of such polynomial
       (4) Fit a BSpline of the raw flux as a function of time
       (5) Normalize the raw flux by the fitted BSpline computed in step (4)
       (6) Bin and interpolate the normalized flux as a function of the arclength
       (7) Divide the raw flux by the piecewise linear interpolation done in step (6)
       (8) Set raw flux as the flux computed in step (7) and repeat
       (9) Multiply back the fitted BSpline

    Parameters
    ----------
    lightcurve : `~lightkurve.lightcurve.LightCurve`
        The light curve object on which the SFF algorithm will be applied.

    Examples
    --------
    >>> lc = LightCurve(time, flux)   # doctest: +SKIP
    >>> corrector = SFFCorrector(lc)   # doctest: +SKIP
    >>> new_lc = corrector.correct(centroid_col, centroid_row)   # doctest: +SKIP
    """
    def __init__(self, lightcurve):
        self.lc = lightcurve

    def correct(self, centroid_col=None, centroid_row=None,
                polyorder=5, niters=3, bins=15, windows=10, sigma_1=3.,
                sigma_2=5., restore_trend=False):
        """Returns a systematics-corrected LightCurve.

        Parameters
        ----------
        centroid_col, centroid_row : array-like, array-like
            Centroid column and row coordinates as a function of time.
            If `None`, then the `centroid_col` and `centroid_row` attributes
            of the `LightCurve` passed to the constructor of this class
            will be used, if present.
        polyorder : int
            Degree of the polynomial which will be used to fit one
            centroid as a function of the other.
        niters : int
            Number of iterations of the aforementioned algorithm.
        bins : int
            Number of bins to be used in step (6) to create the
            piece-wise interpolation of arclength vs flux correction.
        windows : int
            Number of windows to subdivide the data.  The SFF algorithm
            is ran independently in each window.
        sigma_1, sigma_2 : float, float
            Sigma values which will be used to reject outliers
            in steps (6) and (2), respectivelly.
        restore_trend : bool
            If `True`, the long-term trend will be added back into the
            lightcurve.

        Returns
        -------
        corrected_lightcurve : `~lightkurve.lightcurve.LightCurve`
            Returns a corrected light curve.
        """
        # `new_lc` is the object we will return at the end of this function;
        # SFF does not work on cadences with flux=NaN so we remove them here.
        new_lc = self.lc.remove_nans().copy()

        # Input validation
        if centroid_col is None:
            try:
                centroid_col = new_lc.centroid_col
            except AttributeError:
                raise ValueError('`centroid_col` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')
        if centroid_row is None:
            try:
                centroid_row = new_lc.centroid_row
            except AttributeError:
                raise ValueError('`centroid_row` must be passed to `correct()` '
                                 'because it is not a property of the LightCurve.')

        # Split the data into windows
        time = np.array_split(new_lc.time, windows)
        flux = np.array_split(new_lc.flux, windows)
        centroid_col = np.array_split(centroid_col, windows)
        centroid_row = np.array_split(centroid_row, windows)
        self.trend = np.array_split(np.ones(len(new_lc.time)), windows)

        # Apply the correction iteratively
        for n in range(niters):
            # First, fit a spline to capture the long-term varation
            # We don't want to fit the long-term trend because we know
            # that the K2 motion noise is a high-frequency effect.
            tempflux = np.asarray([item for sublist in flux for item in sublist])
            flux_outliers = sigma_clip(data=tempflux, sigma=sigma_1).mask
            self.bspline = self.fit_bspline(new_lc.time[~flux_outliers], tempflux[~flux_outliers])

            # The SFF algorithm is going to be run on each window independently
            for i in range(windows):
                # To make it easier (and more numerically stable) to fit a
                # characteristic polynomial that describes the spacecraft motion,
                # we rotate the centroids to a new coordinate frame in which
                # the dominant direction of motion is aligned with the x-axis.
                self.rot_col, self.rot_row = self.rotate_centroids(centroid_col[i],
                                                                   centroid_row[i])
                # Next, we fit the motion polynomial after removing outliers
                self.outlier_cent = sigma_clip(data=self.rot_col,
                                               sigma=sigma_2).mask
                with warnings.catch_warnings():
                    # ignore warning messages related to polyfit being poorly conditioned
                    warnings.simplefilter("ignore", category=np.RankWarning)
                    coeffs = np.polyfit(self.rot_row[~self.outlier_cent],
                                        self.rot_col[~self.outlier_cent], polyorder)

                self.poly = np.poly1d(coeffs)
                self.polyprime = np.poly1d(coeffs).deriv()

                # Compute the arclength s.  It is the length of the polynomial
                # (fitted above) that describes the typical motion.
                x = np.linspace(np.min(self.rot_row[~self.outlier_cent]),
                                np.max(self.rot_row[~self.outlier_cent]), 10000)
                self.s = np.array([self.arclength(x1=xp, x=x) for xp in self.rot_row])

                # Remove the long-term variation by dividing the flux by the spline
                iter_trend = self.bspline(time[i])
                self.normflux = flux[i] / iter_trend
                self.trend[i] = iter_trend
                # Bin and interpolate normalized flux to capture the dependency
                # of the flux as a function of arclength
                self.interp = self.bin_and_interpolate(self.s, self.normflux, bins,
                                                       sigma=sigma_1)
                # Correct the raw flux
                corrected_flux = self.normflux / self.interp(self.s)
                flux[i] = corrected_flux
                if restore_trend:
                    flux[i] *= self.trend[i]

        new_lc.flux = np.asarray([item for sublist in flux for item in sublist])
        return new_lc

    def rotate_centroids(self, centroid_col, centroid_row):
        """Rotate the coordinate frame of the (col, row) centroids to a new (x,y)
        frame in which the dominant motion of the spacecraft is aligned with
        the x axis.  This makes it easier to fit a characteristic polynomial
        that describes the motion."""
        centroids = np.array([centroid_col, centroid_row])
        _, eig_vecs = linalg.eigh(np.cov(centroids))
        return np.dot(eig_vecs, centroids)

    def _plot_rotated_centroids(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.rot_row[~self.outlier_cent], self.rot_col[~self.outlier_cent],
                'ko', markersize=3)
        ax.plot(self.rot_row[~self.outlier_cent], self.rot_col[~self.outlier_cent],
                'bo', markersize=2)
        ax.plot(self.rot_row[self.outlier_cent], self.rot_col[self.outlier_cent],
                'ko', markersize=3)
        ax.plot(self.rot_row[self.outlier_cent], self.rot_col[self.outlier_cent],
                'ro', markersize=2)
        x = np.linspace(min(self.rot_row), max(self.rot_row), 200)
        ax.plot(x, self.poly(x), '--')
        plt.xlabel("Rotated row centroid")
        plt.ylabel("Rotated column centroid")
        return ax

    def _plot_normflux_arclength(self):
        idx = np.argsort(self.s)
        s_srtd = self.s[idx]
        normflux_srtd = self.normflux[idx]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(s_srtd[~self.outlier_mask], normflux_srtd[~self.outlier_mask],
                'ko', markersize=3)
        ax.plot(s_srtd[~self.outlier_mask], normflux_srtd[~self.outlier_mask],
                'bo', markersize=2)
        ax.plot(s_srtd[self.outlier_mask], normflux_srtd[self.outlier_mask],
                'ko', markersize=3)
        ax.plot(s_srtd[self.outlier_mask], normflux_srtd[self.outlier_mask],
                'ro', markersize=2)
        ax.plot(s_srtd, self.interp(s_srtd), '--')
        plt.xlabel(r"Arclength $(s)$")
        plt.ylabel(r"Flux $(e^{-}s^{-1})$")
        return ax

    def arclength(self, x1, x):
        """Compute the arclength of the polynomial used to fit the centroid
        measurements.

        Parameters
        ----------
        x1 : float
            Upper limit of the integration domain.
        x : ndarray
            Domain at which the arclength integrand is defined.

        Returns
        -------
        arclength : float
            Result of the arclength integral from x[0] to x1.
        """
        mask = x < x1
        return np.trapz(y=np.sqrt(1 + self.polyprime(x[mask]) ** 2), x=x[mask])

    def fit_bspline(self, time, flux, knotspacing=1.5):
        """Returns a `scipy.interpolate.BSpline` object to interpolate flux as a function of time."""
        # By default, bspline knots are placed 1.5 days apart
        knots = np.arange(time[0], time[-1], knotspacing)

        # If the light curve has breaks larger than the spacing between knots,
        # we must remove the knots that fall in the breaks.
        # This is necessary for e.g. K2 Campaigns 0 and 10.
        bad_knots = []
        a = time[:-1][np.diff(time) > knotspacing]  # times marking the start of a gap
        b = time[1:][np.diff(time) > knotspacing]  # times marking the end of a gap
        for a1, b1 in zip(a, b):
            bad = np.where((knots > a1) & (knots < b1))[0][1:-1]
            [bad_knots.append(b) for b in bad]
        good_knots = list(set(list(np.arange(len(knots)))) - set(bad_knots))
        knots = knots[good_knots]

        # Now fit and return the spline
        t, c, k = interpolate.splrep(time, flux, t=knots[1:])
        return interpolate.BSpline(t, c, k)

    def bin_and_interpolate(self, s, normflux, bins, sigma):
        idx = np.argsort(s)
        s_srtd = s[idx]
        normflux_srtd = normflux[idx]

        self.outlier_mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask
        normflux_srtd = normflux_srtd[~self.outlier_mask]
        s_srtd = s_srtd[~self.outlier_mask]

        knots = np.array([np.min(s_srtd)]
                         + [np.median(split) for split in np.array_split(s_srtd, bins)]
                         + [np.max(s_srtd)])
        bin_means = np.array([normflux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(normflux_srtd, bins)]
                             + [normflux_srtd[-1]])
        return interpolate.interp1d(knots, bin_means, bounds_error=False,
                                    fill_value='extrapolate')

    def breakpoints(self, campaign):
        """Return a break point as a function of the campaign number.

        The intention of this function is to implement a smart way to determine
        the boundaries of the windows on which the SFF algorithm is applied
        independently. However, this is not implemented yet in this version.
        """
        raise NotImplementedError()
