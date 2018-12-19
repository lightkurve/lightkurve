from __future__ import division, print_function

import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt



from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

from scipy import interpolate, linalg

log = logging.getLogger(__name__)


def fit_bspline(time, flux, knotspacing=1.5):
    # By default, knots are placed 1.5 days apart
    knots = np.arange(time[0], time[-1], knotspacing)

    # If the light curve has breaks larger than the spacing between knots,
    # we must remove the knots that fall in the breaks.
    # This is necessary for e.g. K2 Campaigns 0 and 10.
    bad_knots = []
    a = time[0:-1][np.diff(time) > knotspacing]
    b = time[1:][np.diff(time) > knotspacing]
    for a1, b1 in zip(a, b):
        bad = np.where((knots > a1) & (knots < b1))[0][1:-1]
        if len(bad_knots) > 0:
            [bad_knots.append(b) for b in bad]
    good_knots = list(set(list(np.arange(len(knots)))) - set(bad_knots))
    knots = knots[good_knots]

    # Now fit and return the spline
    t, c, k = interpolate.splrep(time, flux, t=knots[1:])
    return interpolate.BSpline(t, c, k)


class SFFCorrectorException(Exception):
    """Raised if there is a problem correcting the data."""
    pass


class SFFCorrector(object):
    ''' A generic class for a corrector object. Correctors take the following form

            corrector = lk.CorrectorClass(data)

        The correctors must have at least two functions

            lk.CorrectorClass(data).correct(**options)
            lk.CorrectorClass(data).plot_diagnostic()

        ...And they always return a lightkurve.LightCurve class object.

        lc = lk.CorrectorClass(data).correct(**options)
    '''

    def __init__(self, data, knotspacing=1.5, xmotion=None, ymotion=None):
        self.data = data.copy()#.remove_nans()
        self.knotspacing = knotspacing

        if xmotion is None:
            self.xmotion =self.data.centroid_row
        else:
            self.xmotion = xmotion

        if ymotion is None:
            self.ymotion =self.data.centroid_col
        else:
            self.ymotion = ymotion

        if (~np.isfinite(self.ymotion)).any() | (~np.isfinite(self.xmotion)).any():
            raise SFFCorrectorException('There are nan values in xmotion and ymotion. Please remove these before proceeding.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.RankWarning)
            poly = np.polyfit(self.xmotion, self.ymotion, 5)
        polyprime = np.poly1d(poly).deriv()
        x = np.linspace(self.xmotion.min(), self.xmotion.max(), 10000)
        # THIS IS HECKIN' SLOW
        self.arclength = np.array([self._compute_arclength(x1=np.asarray(xp), x=x, polyprime=polyprime) for xp in self.xmotion])

        self.knotspacing = knotspacing


    def __repr__(self):
        return 'lightkurve.SFFCorrector Class'


    #-----------------
    # Data Quality

    def thrusters(self, sigma=3):
        #rad = (self.data.centroid_col**2 + self.data.centroid_row**2)**0.5
        mask = np.zeros(len(self.data.time), dtype=bool)
        bp = np.where(np.append(10, np.diff(self.data.time)) > 30 * np.median(np.diff(self.data.time)))[0]
        bp = np.append(bp, len(self.data.time))
        if len(bp) <= 2:
            bp = [0, len(self.data.time)//2, len(self.data.time)]

        for b1, b2 in zip(bp[0:-1], bp[1:]):
            _, med, std = sigma_clipped_stats(np.diff(self.arclength[b1:b2]), sigma=sigma, iters=3)
            thrusters = np.zeros(len(self.arclength[b1:b2]), dtype=bool)
            thrusters[1:] |= np.abs(np.diff(self.arclength[b1:b2]) - med) > sigma * std
            thrusters[0:-1] |= np.abs(np.diff(self.arclength[b1:b2]) - med) > sigma * std

            _, med, std = sigma_clipped_stats(self.arclength[b1:b2], sigma=sigma, iters=3)
            thrusters |= np.abs(self.arclength[b1:b2] - med) > sigma * std

#            plt.scatter(self.data.time[b1:b2], self.arclength[b1:b2], s=1)
#            plt.scatter(self.data.time[b1:b2][thrusters], self.arclength[b1:b2][thrusters], s=10, c='b')
            mask[b1:b2] |= thrusters

#        plt.scatter(self.data.time, self.arclength, c='k', s=1)
#        plt.scatter(self.data.time[mask], self.arclength[mask], c='r', s=10)

        return mask

    def outliers(self, sigma=3, niters=5):
        mask = np.zeros(len(self.data.time), dtype=bool)
        for iter in range(niters):
            f = np.copy(self.data.flux)/self.correction
            model = fit_bspline(self.data.time[~mask], f[~mask], self.knotspacing)(self.data.time)
            m1 = np.abs(np.nan_to_num(f - model)) > (sigma * np.nanstd(f - model))
            mask |= convolve(m1, Gaussian1DKernel(2)) > 0.01
        return mask

    def burnin(self, width=20):
        mask = np.append(10, np.diff(self.data.time)) > np.median(np.diff(self.data.time)) * 30
        mask = convolve(mask, Box1DKernel(width), fill_value=0, boundary='fill') > 0.01
        return mask

    def nans(self):
        mask = ~np.isfinite(self.data.flux)
        mask |= ~np.isfinite(self.data.time)
        return mask

    #-----------------
    # Windowing
    def _find_breaks(self, break_tolerance=20, min_size=50):
        """Evenly space a user specified number of windows, given some break points, shift and optimal size.

        Tries to create roughly evenly sized windows, without any small segments. Allows for shifts.
        Takes into account natural break points in the data.

        Parameters
        ----------

        min_size : int
            Minimum size of any given window. Windows will be merged when they drop below this size.
        Returns
        -------

        break_points : list
            Where the data should be broken.
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


    def _bin_and_interpolate(self, s, normflux, bins=10):
        if len(s) == 0:
            return lambda x: []
        normflux_srtd = normflux[np.argsort(s)]
        s_srtd = s[np.argsort(s)]

        # where are we going to spine?
        knots = np.array([np.min(s_srtd)]
                         + [np.median(split) for split in np.array_split(s_srtd, bins)]
                         + [np.max(s_srtd)])

        bin_means = np.array([normflux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(normflux_srtd, bins)]
                             + [normflux_srtd[-1]])
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


    def correct(self, sigma=3, niters=5, windows=20, window_shift=0, bins=10, user_break_points=None, remove_trend=False):

        self.bins = bins
        self.windows = windows
        self.window_shift = window_shift
        self.user_break_points = user_break_points
        self.breakpoints = self._find_breaks()



        # Make a simple first pass.
        l = self._bin_and_interpolate(self.arclength, self.data.flux, bins=self.bins)
        self.correction = l(self.arclength)

        self.data = self.data.remove_nans()
        self.mask = ~self.thrusters() & ~self.outliers() & ~self.burnin() & ~self.nans()
        #self._validate()


        flux = np.copy(self.data.flux)

        for count in np.arange(niters):
            longtermtrends = fit_bspline(self.data.time[self.mask], self.data.flux[self.mask]/self.correction[self.mask], knotspacing=self.knotspacing)(self.data.time)
            for b1, b2 in zip(self.breakpoints[0:-1], self.breakpoints[1:]):
                y = np.copy(flux[b1:b2])
                y /= self.correction[b1:b2]
                y /= longtermtrends[b1:b2]
                x = self.arclength[b1:b2]

                # Mask out bad parts of the window...
                mask = self.mask[b1:b2]
                k = (~sigma_clip(x[mask], sigma=sigma).mask & ~sigma_clip(y[mask], sigma=sigma).mask)
                mask[mask] &= k
                if mask.sum() == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    c1 = self._bin_and_interpolate(x[mask][np.argsort(x[mask])], y[mask][np.argsort(x[mask])])(x)
                c1[~np.isfinite(c1)] = 1
                self.correction[b1:b2] *= c1/np.median(c1[mask])
            self.mask &= ~self.outliers()
        if remove_trend:
            self.correction *= longtermtrends
        return (self.data/self.correction)[~self.burnin() & ~self.thrusters()]

    def _create_plot(self):
        ''' Plotting style for creating a diagnostic...?
        '''
        pass


    def interactive(self, notebook_url='localhost:8888', postprocessing=None):
        """Display an interactive Jupyter Notebook widget to inspect the pixel data.

        The widget will show both the lightcurve and pixel data.  By default,
        the lightcurve shown is obtained by calling the `to_lightcurve()` method,
        unless the user supplies a custom `LightCurve` object.
        This feature requires an optional dependency, bokeh (v0.12.15 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        At this time, this feature only works inside an active Jupyter
        Notebook, and tends to be too slow when more than ~30,000 cadences
        are contained in the TPF (e.g. short cadence data).

        Parameters
        ----------
        lc : LightCurve object
            An optional pre-processed lightcurve object to show.
        notebook_url: str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        max_cadences : int
            Raise a RuntimeError if the number of cadences shown is larger than
            this value. This limit helps keep browsers from becoming unresponsive.
        """
        from .interact import show_SFF_interact_widget
        return show_SFF_interact_widget(self, notebook_url=notebook_url,
                                        postprocessing=postprocessing)
