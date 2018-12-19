'''Module for implementing Self Flat Fielding correction on light curves
'''
from __future__ import division, print_function

import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
from .lightcurve import LightCurve
from .lightcurvefile import LightCurveFile
from .targetpixelfile import TargetPixelFile
from .utils import LightkurveWarning

from scipy import interpolate, linalg
from . import PACKAGEDIR, MPLSTYLE

log = logging.getLogger(__name__)


def fit_bspline(time, flux, knotspacing=1.5, return_knots=False):
    ''' Fit a Bivariate Spline to time/flux data

    Parameters
    ----------
    time : np.ndarray
        Time data to fit
    flux : np.ndaarray
        Flux data to fit
    knot_spacing : float
        The spacing in time at which to set knots for the spline.
        Knots that fall in gaps in time will be removed. Default 1.5
    return_knots : bool
        Whether to return the spline function, or the spline function AND knots (as list).
        Default False.

    Returns
    -------
    func : function
        An interpolated bspline.
    '''
    # By default, knots are placed 1.5 days apart
    knots = np.arange(time[0] + 0.55*knotspacing, time[-1] - 0.55*knotspacing, knotspacing)
    knots = knots[np.asarray([np.min(np.abs(time - k)) < 0.55*knotspacing for k in knots])]
    # Now fit and return the spline
    t, c, k = interpolate.splrep(time, flux, t=knots)
    if return_knots:
        return interpolate.BSpline(t, c, k), knots
    return interpolate.BSpline(t, c, k)


class SFFCorrectorException(Exception):
    """Raised if there is a problem correcting the data."""
    pass


class SFFCorrector(object):
    '''SFF Corrector class.
    '''

    def __init__(self, data, xmotion=None, ymotion=None):
        ''' SFF Corrector Class

        Parameters
        ----------
        data : lightkurve.LightCurve object.
            Light curve object to correct
        xmotion : Optional, np.ndarray of same length as data.flux
            Motion in the x direction. Optional. Will use data.centroid_row by default
        ymotion : Optional, np.ndarray of same length as data.flux
            Motion in the x direction. Optional. Will use data.centroid_col by default
        '''
        # Accepted corrector data types
        acceptable_types = (LightCurveFile,
                            LightCurve,
                            TargetPixelFile)

        if not isinstance(data, acceptable_types):
            raise CorrectorException("`data` is an invalid type.\nData must be one of {}".format(acceptable_types))
        if isinstance(data, LightCurveFile):
            warnings.warn('Passed a LightCurveFile, not a LightCurve. Using SAP_FLUX.', LightkurveWarning)
            self.data = data.SAP_FLUX.remove_nans().normalize().copy()
        elif isinstance(data, TargetPixelFile):
            warnings.warn('Passed a TargetPixelFile, not a LightCurve. Using default aperture.', LightkurveWarning)
            self.data = data.to_lightcurve().remove_nans().normalize().copy()
        else:
            self.data = data.remove_nans().normalize().copy()

        self._is_corrected = False

        if xmotion is None:
            self.xmotion =self.data.centroid_col
        else:
            self.xmotion = xmotion

        if ymotion is None:
            self.ymotion =self.data.centroid_row
        else:
            self.ymotion = ymotion

        if np.any([(len(self.xmotion) != len(self.data.flux)), (len(self.ymotion) != len(self.data.flux))]):
            raise SFFCorrectorException('Motion vectors are not the same shape as the input light curve.')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)
            poly = np.polyfit(self.xmotion[~self.nans()], self.ymotion[~self.nans()], 5)
        polyprime = np.poly1d(poly).deriv()
        x = np.linspace(self.xmotion[~self.nans()].min(), self.xmotion[~self.nans()].max(), 10000)
        self.arclength = np.array([self._compute_arclength(x1=np.asarray(xp), x=x, polyprime=polyprime) for xp in self.xmotion])
        self.mask = ~self.motion_outliers() & ~self.burnin() & ~self.nans()

    def __repr__(self):
        return 'lightkurve.SFFCorrector Class'


    #-----------------
    # Data Quality

    def motion_outliers(self, sigma=3):
        ''' Returns a boolean mask, True where there are outliers in motion.

        Note: splits light curve into two halves, as there are two different motion properties
        '''
        #rad = (self.data.centroid_col**2 + self.data.centroid_row**2)**0.5
        mask = np.zeros(len(self.data.time), dtype=bool)
        bp = np.where(np.append(10, np.diff(self.data.time)) > 30 * np.median(np.diff(self.data.time)))[0]
        bp = np.append(bp, len(self.data.time))
        if len(bp) <= 2:
            bp = [0, len(self.data.time)//2, len(self.data.time)]

        for b1, b2 in zip(bp[0:-1], bp[1:]):
            _, med, std = sigma_clipped_stats(np.diff(self.arclength[b1:b2]), sigma=sigma, iters=3)
            motion_outliers = np.zeros(len(self.arclength[b1:b2]), dtype=bool)
            _, med, std = sigma_clipped_stats(self.arclength[b1:b2], sigma=sigma, iters=3)
            motion_outliers |= np.abs(self.arclength[b1:b2] - med) > sigma * std
            mask[b1:b2] |= motion_outliers
        return mask

    def outliers(self, sigma=3, niters=5):
        ''' Returns a boolean mask, True where there are outliers in flux.
        '''
        mask = np.zeros(len(self.data.time), dtype=bool)
        for iter in range(niters):
            f = np.copy(self.data.flux)/self.correction
            model = fit_bspline(self.data.time[~mask], f[~mask], self.knotspacing)(self.data.time)
            m1 = np.abs(np.nan_to_num(f - model)) > (sigma * np.nanstd(f - model))
            mask |= convolve(m1, Gaussian1DKernel(2)) > 0.01
        return mask

    def burnin(self, width=20):
        ''' Returns a boolean mask, True where there is data after a long break.
        This masks where there may be "burn in" at the beginning of campaigns.
        '''
        mask = np.append(10, np.diff(self.data.time)) > np.median(np.diff(self.data.time)) * 30
        mask = convolve(mask, Box1DKernel(width), fill_value=0, boundary='fill') > 0.01
        return mask

    def nans(self):
        ''' Returns a boolean mask, True where there are nan values in the data.
        '''
        mask = ~np.isfinite(self.data.flux)
        mask |= ~np.isfinite(self.data.time)
        mask |= ~np.isfinite(self.xmotion)
        mask |= ~np.isfinite(self.ymotion)

        return mask

    #-----------------
    # Windowing
    def _find_breaks(self, break_tolerance=20, min_size=50):
        """Evenly space a user specified number of windows, given some break points, shift and optimal size.

        Tries to create roughly evenly sized windows, without any small segments. Allows for shifts.
        Takes into account natural break points in the data.

        Parameters
        ----------
        break_tolerance : int
            Breaks in data longer than this number of points will be considered to be new sections
        min_size: int
            Segments shorter than this length will not be allowed, and will be merged with the preceeding segment.
        Returns
        -------
        break_points : list
            Points where the data should be broken.
        """

        # Number of points in a window
        dw = len(self.data.time) // self.windows
        window_shift = (self.window_shift % dw)

        # Where there are significant breaks in self.data.time
        breakpoints = np.where(np.diff(self.data.time)/np.median(np.diff(self.data.time)) > break_tolerance)[0] + 1
        breakpoints = np.append(np.append(0, breakpoints), len(self.data.time))

        # Add in user break points
        if self.user_break_points is not None:
            breakpoints = np.sort(np.append(breakpoints, np.asarray(self.user_break_points)))

        result = np.zeros(0, dtype=int)
        for b1, b2 in zip(breakpoints[0:-1], breakpoints[1:]):
            bp1 = np.arange(b1, b2, dw) + window_shift
            bp1 = np.append(b1, bp1)
            bp1 = np.append(bp1, b2)
            bp1 = np.sort(bp1)
            bp1 = bp1[(bp1 >= b1) & (bp1 <= b2)]
            diff = np.diff(bp1)
            if (diff < min_size).any():
                while (diff < min_size).any():
                    bad = np.where(diff < min_size)[0][0] + 1
                    if bad == len(bp1) - 1:
                        break
                    bp1 = bp1[~np.in1d(np.arange(len(bp1)), bad)]
                    diff = np.diff(bp1)
            result = np.append(result, bp1)
        return np.sort(np.unique(result))
    #-----------------


    def _bin_and_interpolate(self, arclength, flux, bins=10):
        ''' Bins the arclength vs. flux data and fits an interpolated curve

        Parameters
        ----------
        s : np.ndarray
            Arclength
        flux : np.ndarray
            Flux
        bins : int
            How agressively to bin the data before fitting the curve. Fewer bins will give a less
            flexible curve.

        Returns
        -------
        func : function
            Interpolated relationship between arclength and flux.
        '''
        if len(arclength) == 0:
            return lambda x: []
        flux_srtd = flux[np.argsort(arclength)]
        arclength_srtd = arclength[np.argsort(arclength)]

        # where are we going to spine?
        knots = np.array([np.min(arclength_srtd)]
                         + [np.median(split) for split in np.array_split(arclength_srtd, bins)]
                         + [np.max(arclength_srtd)])

        bin_means = np.array([flux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(flux_srtd, bins)]
                             + [flux_srtd[-1]])
        return interpolate.interp1d(knots, bin_means, bounds_error=False,
                                    fill_value='extrapolate')


    def _compute_arclength(self, x1, x, polyprime, index=0):
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
        y = np.sqrt(1 + polyprime(x[mask]) ** 2)
        return np.trapz(y=y, x=x[mask])


    def correct(self, windows=20, window_shift=0, bins=10, user_break_points=None, knotspacing=1.5, sigma=3, niters=5, remove_trend=False):
        '''Corrects the input data using SFF method.

        Parameters
        ----------
        windows : int
            Number of windows to split the data into. Some windows may be removed or merged.
        window_shift : int
            Number of points to shift the data in time
        bins : int
            Number of bins to use when fitting arclength. A smaller number of bins is less flexible,
            but more robust.
        user_break_points : list
            A list of input break points that the user specifies. Data will be broken at these points,
            in addition to the computed windows. Default is None.
        knotspacing : float
            How far apart in time to space knots when using a bspline. A larger knot spacing will fit
            longer trends, a smaller spacing will fit shorter trends.
        sigma : int
            How many sigma outliers should be removed from fitting the arclength trend.
        niters : int
            How many iterations to perform the correction. Default 5
        remove_trend : bool
            Whether to return a corrected light curve with the long term trends removed.

        Returns
        -------
        corrected_lc : lightkurve.LightCurve
            Light curve with the correction divded out, and outliers in motion and burnin removed.
        '''
        self.bins = bins
        self.windows = windows
        self.window_shift = window_shift
        self.user_break_points = user_break_points
        self.knotspacing = knotspacing
        self.breakpoints = self._find_breaks()

        # Make a simple first pass.
        l = self._bin_and_interpolate(self.arclength, self.data.flux, bins=self.bins)
        self.correction = l(self.arclength)
        self.mask &= ~self.outliers(sigma=sigma)


        flux = np.copy(self.data.flux)
        for count in np.arange(niters):
            longtermtrends = fit_bspline(self.data.time[self.mask], self.data.flux[self.mask]/self.correction[self.mask], knotspacing=self.knotspacing)(self.data.time)
            for b1, b2 in zip(self.breakpoints[0:-1], self.breakpoints[1:]):

                y = np.copy(flux[b1:b2])
                y /= self.correction[b1:b2]
                y /= longtermtrends[b1:b2]
                x = self.arclength[b1:b2]

                # Mask out bad parts of the window...
                mask = np.copy(self.mask[b1:b2])
                k = (~sigma_clip(x[mask], sigma=sigma).mask & ~sigma_clip(y[mask], sigma=sigma).mask)
                mask[mask] &= k
                if mask.sum() == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    c1 = self._bin_and_interpolate(x[mask][np.argsort(x[mask])], y[mask][np.argsort(x[mask])])(x)
                c1[~np.isfinite(c1)] = 1
                self.correction[b1:b2] *= c1/np.median(c1[mask])
            self.mask &= ~self.outliers(sigma=sigma)



        self._is_corrected = True
        self.corrected = (self.data/self.correction)
        if remove_trend:
            return (self.corrected/longtermtrends)[~self.burnin() & ~self.motion_outliers()]
        return (self.corrected)[~self.burnin() & ~self.motion_outliers()]

    def plot_diagnostic(self):
        '''Plot a diagnostic of the performance of the correct method.

        Returns
        -------
        fig : mpl.pyplot.figure
            Diagnostic figure for SFF correct method.
        '''

        if not self._is_corrected:
            raise SFFCorrectorException('Please run correction with the correct method before trying to diagnose.')
        with plt.style.context(MPLSTYLE):
            fig = plt.figure(figsize=(15, 7))
            ax = plt.subplot2grid((2,4), (0, 0), colspan=3)
            self.data.scatter(ax=ax, label='Original', c='k', normalize=False)
            self.data[self.motion_outliers()].scatter(ax=ax, label='Motion Outliers', c='r', normalize=False, s=10)
            self.data[self.burnin()].scatter(ax=ax, label='Burn In', c='b', normalize=False, s=10)


            longtermtrends, knots = fit_bspline(self.data.time[self.mask], self.data.flux[self.mask]/self.correction[self.mask], knotspacing=self.knotspacing, return_knots=True)
            ax.plot(self.data.time, longtermtrends(self.data.time), lw=4, alpha=0.4, c='lime', label='B-Spline')
            ax.scatter(knots, longtermtrends(knots), c='lime', s=20, edgecolor='k', lw=0.5)

            ax.set_xlabel('')
            ax.legend()
            xlims, ylims = ax.get_xlim(), ax.get_ylim()
            ax = plt.subplot2grid((2,4), (1, 0), colspan=3)
            self.corrected.plot(ax=ax, label='Corrected', normalize=False)
            self.corrected[self.outliers()].scatter(ax=ax, label='Outliers (Not used for corection)', c='r', normalize=False)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            ax = plt.subplot2grid((2,4), (0,3))
            ax.scatter(self.arclength[~self.mask], (self.data.flux/longtermtrends(self.data.time))[~self.mask], c='r', label='Masked')
            ax.scatter(self.arclength[self.mask], (self.data.flux/longtermtrends(self.data.time))[self.mask], c='k', s=1, label='True Arclength')

            for b1, b2 in zip(self.breakpoints[0:-1], self.breakpoints[1:]):
                x, y = self.arclength[b1:b2], self.correction[b1:b2]
                ax.plot(x[np.argsort(x)], y[np.argsort(x)], c='orange')
            ax.plot(self.arclength[b1:b2], self.correction[b1:b2], c='orange', label='Correction')


            ax.set_ylabel('Flux')
            ax.set_xlabel('Arclength')
            ax.set_ylim(ylims)
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.legend()

        return fig


    def interactive(self, notebook_url='localhost:8888', postprocessing=None):
        """Display an interactive Jupyter Notebook widget to interactively correct with SFF.

        Parameters
        ----------
        notebook_url: str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        post_processing : function
            A function that will be applied to the corrected light curve before displaying.
        """
        from .interact import show_SFF_interact_widget
        return show_SFF_interact_widget(self, notebook_url=notebook_url,
                                        postprocessing=postprocessing)
