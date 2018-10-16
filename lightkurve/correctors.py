"""Defines KeplerCBVCorrector and SFFCorrector."""

from __future__ import division, print_function

import logging
import requests
import warnings

from bs4 import BeautifulSoup
from tqdm import tqdm
from copy import deepcopy

from astropy.convolution import convolve, Box1DKernel

import oktopus
import numpy as np
from scipy import linalg, interpolate
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from astropy.io import fits as pyfits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.convolution import convolve, Gaussian1DKernel

from scipy.stats import chisquare
from scipy.optimize import minimize

from .utils import channel_to_module_output
from .lightcurve import LightCurve
from .lightcurvefile import KeplerLightCurveFile

log = logging.getLogger(__name__)
import celerite

__all__ = ['KeplerCBVCorrector', 'SFFCorrector']


class MaskError(Exception):
    """Raised if there is a problem masking the data."""
    pass


class GPCorrector(object):
    r'''Remove long term trends by fitting a Gaussian Process using celerite

    Uses `celerite` and a Matern 3/2 Kernel and a Jitter term to model the
    long term noise.

    Attributes
    ----------
    lc : LightCurve object
        A light curve of the target
    mask : boolean mask with the same length as lc.flux (optional)
        Mask to avoid fitting to signals where time scales should be preserved.
        e.g. if the light curve has transits, it is strongly recommended to mask
        these out.
        Points where mask is False will be removed from the fit.
    '''

    def find_breaks(self):
        '''Find breaks in data collection that are more than the break_tolerance.
        '''
        breaks = np.where(self.dt > self.break_tolerance * np.nanmin(self.dt))[0] + 1
        breaks = np.append([0], breaks)
        breaks = np.append(breaks, len(self.lc.time))
        return breaks

    def __init__(self, lc):
        if np.any([np.isnan(lc.flux), np.isnan(lc.time), np.isnan(lc.flux_err)]):
            raise ValueError('Light curve contains NaN values.')
        if np.nansum(lc.normalize().flux - lc.flux) != 0:
            raise ValueError('Flux is unnormalized.')
        self.lc = lc
        self.break_tolerance = 10
        self.mask = np.ones(len(lc.flux), dtype=bool)
        self.dt = self.lc.time[1:] - self.lc.time[0:-1]
        self.dt_min = np.nanmin(self.dt)
        self.breaks = self.find_breaks()

    def initialize(self, timescale=None, bounds=None, initial=None):
        '''Returns a corrected lightcurve which is a copy of lc.

        Parameters
        ----------
        timescale : float or None
            If not None, sets the lower bound for optimization of the GP. Will
            not fit trends on timescales near to or shorter than this timescale.
        '''
        # Set up the kernels and bounds for celerite
        sigma = self.lc[self.mask].flatten().flux.std()
        jittersigma = self.lc[self.mask].flatten().flux.std()
        if timescale is not None:
            if bounds is None:
                bounds = [(np.log(0.1*sigma), np.log(10*sigma)),
                          (np.log(timescale/2), np.log(timescale*10))]
            log_rho = np.log(timescale)
        else:
            if bounds is None:
                bounds = [(-15, 15), (-15, 15)]
            log_rho = -np.log(0.26)
        if initial is None:
            kernel = celerite.terms.Matern32Term(log_sigma=np.log(sigma),
                                                 log_rho=log_rho, bounds=bounds)
        else:
            kernel = celerite.terms.Matern32Term(log_sigma=initial[0],
                                                 log_rho=initial[1], bounds=bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=np.log(jittersigma))

        gp = celerite.GP(kernel, mean=1, fit_mean=True)
        self.initial_params = gp.get_parameter_dict()
        gp.compute(self.lc[self.mask].time - self.lc.time[0], self.lc[self.mask].flux_err)

        # Define a log liklihood to minimize
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        # Optimize the GP with scipy.minimize
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params,
                        method="TNC", bounds=bounds, args=(self.lc[self.mask].flux, gp))
        self.soln = soln
        gp.set_parameter_vector(soln.x)
        self.best_fit_params = gp.get_parameter_dict()
        self.gp = gp

    def correct(self, timescale=2, iters=2, sigma=3, return_trend=False,
                bounds=None, initial=None, mask=None, mask_burn=False):
        '''Returns a lightcurve corrected by a GP.

        Runs the GP for times, split into different chunks based on gaps in
        data collection.

        Parameters
        ----------
        timescale : float (default 0.25 days)
            The time scale at which the user wishes to preserve signals.
        iters : int
            Number of times to iterate the correction. Each iteration successively
            masks out data that is poorly fit by the GP.
        return_trend : bool (default False)
            If True, returns a LightCurve object with the correction. If False,
            applies the correction to a copy of the original lc.s

        Returns
        -------
        corrected_lc : lightkurve.LightCurve object
            The corrected light curve with propagated errors
        flat : lightkurve.LightCurve object (optional)
            If return_trend is True, will also return the best fit trend.
        '''
        # We need to know how many points the `time_scale` corresponds to.
        if mask is None:
            mask = np.ones(len(self.lc.flux), dtype=bool)
        if mask_burn is True:
            mask &= ~np.in1d(np.arange(0, len(self.lc.time)), np.asarray(
                [np.arange(b, b+20) for b in self.breaks]).ravel())
        # Iteratively sigma clip the data
        for iter in range(iters):
            # In the next run, include this new mask.
            self.mask &= mask
            if 100.*np.nansum(~self.mask)/len(self.mask) > 80:
                raise MaskError('Too large of a fraction of data is masked ({}%).'
                                'Consider running fewer iterations, with a larger '
                                'sigma tolerance or a shorter timescale.'
                                ''.format(int(100.*np.nansum(~self.mask)/len(self.mask))))
            # Start the GP, find the maximum liklihood solution
            self.initialize(timescale=timescale, bounds=bounds, initial=initial)
            flat = LightCurve(time=self.lc.time, flux=self.lc.flux*0, flux_err=self.lc.flux_err*0)
            # Run over each 'portion' of the data, defined where breaks are greater
            # than break_tolerance
            for idx in tqdm(range(len(self.breaks)-1), desc='Iteration {} ({}% Masked)'.format(iter + 1, int(100.*np.nansum(~self.mask)/len(self.mask)))):
                l = self.lc[self.breaks[idx]:self.breaks[idx+1]]
                x = l.time - self.lc.time[0]
                # Calculate the best fit mean and errors
                pred_mean, pred_var = self.gp.predict(self.lc[self.mask].flux, x, return_var=True)
                pred_mean = pred_mean + self.gp.get_parameter_dict()['mean:value'] - 1
                pred_std = np.sqrt(pred_var)
                flat.flux[self.breaks[idx]:self.breaks[idx+1]] = pred_mean - 1
                flat.flux_err[self.breaks[idx]:self.breaks[idx+1]] = pred_std

            # Mask out any data that is different from the model by `sigma`
            corr = np.abs(self.lc.flux - flat.flux)
            corr -= np.nanmedian(corr)
            mask = corr < sigma * ((self.lc.flux_err**2 + (flat.flux_err)**2)**0.5)
#            mask = (convolve(mask, Box1DKernel(boxwidth)) == 1)

        # Return the best fit trend.
        flat.flux -= np.nanmedian(flat.flux)
        corrected_lc = deepcopy(self.lc)
        corrected_lc.flux -= flat.flux
        corrected_lc.flux_err = (self.lc.flux_err**2 + (flat.flux_err)**2)**0.5
        if return_trend:
            flat.flux += 1
            return corrected_lc, flat
        return corrected_lc


class KeplerCBVCorrector(object):
    r"""Remove systematic trends from Kepler light curves by fitting
    cotrending basis vectors.

    .. math::

        \arg \min_{\bm{\theta} \in \Theta} \sum_{t}|f_{SAP}(t) - \sum_{j=1}^{n}\theta_j v_{j}(t)|^p, p>0, p \in \mathbb{R}

    Attributes
    ----------
    lc_file : KeplerLightCurveFile object or str
        An instance from KeplerLightCurveFile or a path for the .fits
        file of a NASA's Kepler/K2 light curve.
    likelihood : oktopus.Likelihood subclass
        A class that describes a cost function.
        The default is :class:`oktopus.LaplacianLikelihood`, which is tantamount
        to the L1 norm.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lightkurve import KeplerCBVCorrector, KeplerLightCurveFile
    >>> fn = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
    ...       "0084/008462852/kplr008462852-2011073133259_llc.fits") # doctest: +SKIP
    >>> cbv = KeplerCBVCorrector(fn) # doctest: +SKIP
    Downloading https://archive.stsci.edu/missions/kepler/lightcurves/0084/008462852/kplr008462852-2011073133259_llc.fits [Done]
    >>> cbv_lc = cbv.correct() # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/cbv/kplr2011073133259-q08-d25_lcbv.fits [Done]
    >>> sap_lc = KeplerLightCurveFile(fn).SAP_FLUX # doctest: +SKIP
    >>> plt.plot(sap_lc.time, sap_lc.flux, 'x', markersize=1, label='SAP_FLUX') # doctest: +SKIP
    >>> plt.plot(cbv_lc.time, cbv_lc.flux, 'o', markersize=1, label='CBV_FLUX') # doctest: +SKIP
    >>> plt.legend() # doctest: +SKIP
    """

    def __init__(self, lc_file, likelihood=oktopus.LaplacianLikelihood,
                 prior=oktopus.LaplacianPrior):
        self.lc_file = lc_file
        self.likelihood = likelihood
        self.prior = prior
        self._ncbvs = 16  # number of cbvs for Kepler/K2

        if self.lc_file.mission == 'Kepler':
            self.cbv_base_url = "http://archive.stsci.edu/missions/kepler/cbv/"
        elif self.lc_file.mission == 'K2':
            self.cbv_base_url = "http://archive.stsci.edu/missions/k2/cbv/"

    @property
    def lc_file(self):
        return self._lc_file

    @lc_file.setter
    def lc_file(self, value):
        # this enables `lc_file` to be either a string
        # or an object from KeplerLightCurveFile
        if isinstance(value, str):
            self._lc_file = KeplerLightCurveFile(value)
        elif isinstance(value, KeplerLightCurveFile):
            self._lc_file = value
        else:
            raise ValueError("lc_file must be either a string or a"
                             " KeplerLightCurveFile instance, got {}.".format(value))

    @property
    def coeffs(self):
        """
        Returns the fitted coefficients.
        """
        return self._coeffs

    @property
    def opt_result(self):
        """
        Returns the result of the optimization process.
        """
        return self._opt_result

    def _get_cbv_data(self, cbvs=[1, 2]):
        '''Gets the CBV data for a channel and module
        '''
        module, output = channel_to_module_output(self.lc_file.channel)
        cbv_file = pyfits.open(self.get_cbv_url())
        cbv_data = cbv_file['MODOUT_{0}_{1}'.format(module, output)].data
        time = cbv_file['MODOUT_{0}_{1}'.format(
            module, output)].data['TIME_MJD'][self.lc_file.quality_mask]
        cbv_array = []
        for i in cbvs:
            cbv_array.append(cbv_data.field('VECTOR_{}'.format(i))[self.lc_file.quality_mask])
        cbv_array = np.asarray(cbv_array)
        return cbv_array, time

    def correct(self, cbvs=[1, 2], method='powell', options={}):
        """
        Correct the SAP_FLUX by fitting a number of cotrending basis vectors
        `cbvs`.

        Parameters
        ----------
        cbvs : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
        method : str
            Numerical optimization method. See scipy.optimize.minimize for the
            full list of methods.
        options : dict
            Dictionary of options to be passed to scipy.optimize.minimize.
        """
        cbv_array, _ = self._get_cbv_data(cbvs)

        sap_lc = self.lc_file.SAP_FLUX
        median_sap_flux = np.nanmedian(sap_lc.flux)
        norm_sap_flux = sap_lc.flux / median_sap_flux - 1
        norm_err_sap_flux = sap_lc.flux_err / median_sap_flux

        def mean_model(*theta):
            coeffs = np.asarray(theta)
            return np.dot(coeffs, cbv_array)

        prior = self.prior(mean=np.zeros(len(cbvs)), var=16.)
        likelihood = self.likelihood(data=norm_sap_flux, mean=mean_model,
                                     var=norm_err_sap_flux)
        x0 = likelihood.fit(x0=prior.mean, method=method, options=options).x
        posterior = oktopus.Posterior(likelihood=likelihood, prior=prior)

        self._opt_result = posterior.fit(x0=x0, method=method,
                                         options=options)
        self._coeffs = self._opt_result.x
        flux_hat = sap_lc.flux - median_sap_flux * mean_model(self._coeffs)
        return LightCurve(time=sap_lc.time, flux=flux_hat.reshape(-1),
                          flux_err=sap_lc.flux_err)

    def get_cbvs_list(self):
        """Returns the subsequence of subsequent CBVs that maximizes
        Bayes' factor [1]_.

        Returns
        -------
        cbv_list : list
            Subsequence of subsequent CBVs that maximizes the Bayes' factor.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Bayes_factor
        """

        self.bayes_factor, cost = [], []  # bayes_factor here is actually the
        # negative log of the bayes factor
        self.correct(cbvs=[1], options={'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 2000})
        cost.append(self.opt_result.fun)
        for n in range(2, self._ncbvs+1):
            cbv_list = list(range(1, n+1))
            self.correct(cbv_list, options={'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 2000})
            cost.append(self.opt_result.fun)
            # cost is the negative log of the posterior evaluated at the
            # Maximum A Posterior Probability (MAP) estimator
            self.bayes_factor.append((cost[n-2] - cost[n-1]))
            # so cost[n-2] - cost[n-1] = -log(p1) + log(p2) = log(p2/p1)
            # where p1 is the posterior probability (evaluated at the MAP)
            # for the model with n-2 cbvs and p2 is the posterior probability
            # also evaluated at the MAP for the model with n-1 cbvs
        k = np.argmin(self.bayes_factor)
        # transform to get the actual Bayes factor
        self.bayes_factor = np.exp(-np.array(self.bayes_factor))
        # the k+2 here comes from the fact that Python indexes begin
        # from 0 and we count CBVs starting from 1 and also
        # note that range(1, k) equals the interval [1, k), which excludes k.
        return list(range(1, k+2))

    def get_cbv_url(self):
        # gets the html page and finds all references to 'a' tag
        # keeps the ones for which 'href' ends with 'fits'
        # this might slow things down in case the user wants to fit 1e3 stars
        soup = BeautifulSoup(requests.get(self.cbv_base_url).text, 'html.parser')
        cbv_files = [fn['href'] for fn in soup.find_all('a') if fn['href'].endswith('fits')]

        if self.lc_file.mission == 'Kepler':
            if self.lc_file.quarter < 10:
                quarter = 'q0' + str(self.lc_file.quarter)
            else:
                quarter = 'q' + str(self.lc_file.quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif self.lc_file.mission == 'K2':
            if self.lc_file.campaign <= 8:
                campaign = 'c0' + str(self.lc_file.campaign)
            else:
                campaign = 'c' + str(self.lc_file.campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break

        return self.cbv_base_url + cbv_file

    def plot_cbvs(self, cbvs=[1, 2], ax=None):
        '''Plot the CBVs for a given list of CBVs

        Parameters
        ----------
        cbvs : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
        ax : matplotlib.pyplot.Axes.AxesSubplot
            Matplotlib axis object. If `None`, one will be generated.

        Returns
        -------
        ax : matplotlib.pyplot.Axes.AxesSubplot
            Matplotlib axis object
        '''
        if ax is None:
            _, ax = plt.subplots(1)
        cbv_array, time = self._get_cbv_data(cbvs)
        for idx, cbv in enumerate(cbv_array):
            ax.plot(time, cbv+idx/10., label='{}'.format(idx + 1))
        ax.set_yticks([])
        ax.set_xlabel('Time (MJD)')
        module, output = channel_to_module_output(self.lc_file.channel)
        if self.lc_file.mission == 'Kepler':
            ax.set_title('Kepler CBVs (Module : {}, Output : {}, Quarter : {})'.format(
                module, output, self.lc_file.quarter))
        elif self.lc_file.mission == 'K2':
            ax.set_title('K2 CBVs (Module : {}, Output : {}, Campaign : {})'.format(
                module, output, self.lc_file.campaign))
        ax.grid(':', alpha=0.3)
        ax.legend()
        return ax


class SFFError(Exception):
    """Raised if there is a problem accessing data."""
    pass

class SFFCorrector(object):
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
    """

    def __init__(self):
        self.fig = None
        self.axs = None
        pass


    def stitch(self, x, y, break_points, knotspacing=1.5, sigma=3, mask=None):
        '''Stitches a light curve back together
        '''
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        def _build_offsets(x, y, break_points, mask, knotspacing=1.5, sigma=3):

            offsets = np.zeros(len(break_points) + 1)
            for i in range(len(break_points) - 1):
                # Cut out two segments
                if i == len(break_points) - 2:
                    data = np.copy(y[break_points[i]:])
                    time = x[break_points[i]:]
                    cutmask = mask[break_points[i]:]
                else:
                    data = np.copy(y[break_points[i]:break_points[i+2]])
                    time = x[break_points[i]:break_points[i+2]]
                    cutmask = mask[break_points[i]:break_points[i+2]]
                b = break_points[i+1] - break_points[i]

                # Check if there is a gap LONGER than the knotspacing at the breakpoint
                if np.any(np.diff(time[b-2:b+3]) > knotspacing):
                    # If there is a long gap...you really can't do anything.
                    # So do nothing.
                    continue

                # Reject some outliers
                mask_a = _outlier_mask(time[:b], data[:b], knotspacing=knotspacing, sigma=sigma)
                mask_b = _outlier_mask(time[b:], data[b:], knotspacing=knotspacing, sigma=sigma)
                outliers = np.append(mask_a, mask_b)
                outliers |= ~np.isfinite(data)
                outliers |= ~cutmask

                # Find the offset
                def func(*params):
                    data1 = np.copy(data)
                    data1[b:] += params[0]
                    model = self.fit_bspline(time[~outliers], data1[~outliers], knotspacing=knotspacing)(time[~outliers])
                    return(np.nansum((data1[~outliers] - model)**2))
                r = minimize(func, [0], method='L-BFGS-B')
                if not r.success:
                    r.x[0] = 0

                data[b:] += r.x[0]

                # Store the offset
                offsets[i+1:] += r.x[0]
            return offsets

        def _reconstruct(y, offsets, break_points):
            result = np.copy(y)
            for o, b1, b2 in zip(offsets, break_points[:-1], break_points[1:]):
                result[b1:b2] += o
            result[b2:] += offsets[-1]
            result /= np.nanmedian(result)
            return result

        def _outlier_mask(x, y, knotspacing=1.5, sigma=3):
            model = self.fit_bspline(x[np.isfinite(y)], y[np.isfinite(y)], knotspacing)(x)
            mask = np.abs(np.nan_to_num(y - model)) > (sigma * np.nanstd(y - model))
            mask = convolve(mask, Gaussian1DKernel(2)) > 0.01
            return mask

        data = np.copy(y)
        outliers = _outlier_mask(x, y, knotspacing=knotspacing, sigma=sigma)
        data[outliers] *= np.nan

        offsets = _build_offsets(x, data, break_points=break_points, mask=mask, knotspacing=knotspacing)
        result = _reconstruct(y, offsets=offsets, break_points=break_points)
        return result

    def build_window_positions(self, origtime, windows=20, window_shift=0, min_size=50, extra_break_points=None, break_tolerance=5):
        """Evenly space a user specified number of windows, given some break points, shift and optimal size.

        Tries to create roughly evenly sized windows, without any small segments. Allows for shifts.
        Takes into account natural break points in the data.

        Parameters
        ----------

        windows : int
            Number of windows to split into
        window_shift : int
            How much to shift the windows by. Can be negative.
        min_size : int
            Minimum size of any given window. Windows will be merged when they drop below this size.
        extra_break_points : np.ndarray of ints
            Code will try to find break points. Use extra_break_points to specify any additional break points the user requires.

        Returns
        -------

        time : list of np.ndarrays
            origtime, split into into optimal segments.
        """

        dw = len(origtime) // windows
        if min_size > dw:
            min_size = dw

        window_shift = (window_shift % dw)

        natural_breakpoints = np.where(np.diff(origtime)/ np.median(np.diff(origtime)) > break_tolerance)[0]
        natural_breakpoints = np.append(np.append(0, natural_breakpoints), len(origtime))
        # Add in some UNnatural ones
        if extra_break_points is not None:
            natural_breakpoints = np.append(natural_breakpoints, np.asarray(extra_break_points))

        if (np.diff(natural_breakpoints) < dw).any():
            while (np.diff(natural_breakpoints) < dw).any():
                bad = np.where(np.diff(natural_breakpoints) < dw)[0]
                natural_breakpoints = natural_breakpoints[np.in1d(np.arange(len(natural_breakpoints)), bad)]
            natural_breakpoints = np.append(np.append(0, natural_breakpoints), len(origtime))

        natural_breakpoints = np.sort(natural_breakpoints)
        time = []
        for i in range(len(natural_breakpoints) - 1):
            # N Points in Segment
            npoints = len(origtime[natural_breakpoints[i]:natural_breakpoints[i+1]])
            # Number of bins needed
            bins = np.max([int(np.round(npoints / dw)), 1])
            # Make them even
            dw1 = npoints // bins
            # Split em
            array =  (np.arange(bins) * dw) + window_shift
            if array[0] < 0:
                array = array[1:]
            temp_time = np.split(origtime[natural_breakpoints[i]:natural_breakpoints[i+1]], array)
            idx = 0

            while idx < len(temp_time):
                t = temp_time[idx]
                if len(t) == 0:
                    idx += 1
                    continue

                if len(t) < min_size:
                    # If it's the last one
                    if (idx == len(temp_time) - 1):
                        t = np.append(time[-1], t)
                        del time[-1]
                    else:
                        t = np.append(t, temp_time[idx + 1])
                        idx += 1

                time.append(t)
                idx += 1
        return time

    def correct(self, origtime, origflux, origcentroid_col, origcentroid_row, bins=15, windows=1,
                            niters=3, sigma_1=3., knotspacing=1.5, window_shift=0,
                            sigma_2=5., restore_trend=False, plot=False, correct_thrusters=True):

        self.windows = windows
        self.bins = bins
        self.knotspacing = knotspacing
        self.nw = niters

        alltime = np.copy(origtime)
        allflux = np.copy(origflux)

        # Get thruster firings
        if correct_thrusters:
            rad = (origcentroid_col**2 + origcentroid_row**2)**0.5
            _, med, std = sigma_clipped_stats(np.diff(rad), sigma=sigma_1, iters=3)
            thrusters = np.ones(len(origcentroid_col), dtype=bool)
            thrusters[1:] &= np.abs(np.diff(rad)) > sigma_2 * std
        else:
            thrusters = np.zeros(len(alltime), dtype=bool)

        # Split up time, flux, column and row into windows
        break_points = self.build_window_positions(origtime, self.windows, window_shift)
        break_points = [np.where(origtime == b[0])[0][0] + 1 for b in break_points]

        time = self.build_window_positions(origtime, self.windows, window_shift)
        flux, centroid_col, centroid_row, thrust = [], [], [], []
        for t in time:
            ok = np.in1d(origtime, t)
            flux.append(origflux[ok])
            centroid_col.append(origcentroid_col[ok])
            centroid_row.append(origcentroid_row[ok])
            thrust.append(thrusters[ok])
        self.windows = len(time)

        # To make it easier (and more numerically stable) to fit a
        # characteristic polynomial that describes the spacecraft motion,
        # we rotate the centroids to a new coordinate frame in which
        # the dominant direction of motion is aligned with the x-axis.

        # Here we are going to calculate the arclength for each point
        self.rot_col, self.rot_row, self.poly, self.polyprime = [None] * self.windows, [None] * self.windows, [None] * self.windows, [None] * self.windows
        self.coeffs, self.s = [None] * self.windows, [None] * self.windows
        for i in range(self.windows):
            self.rot_col[i], self.rot_row[i] = self.rotate_centroids(centroid_col[i],
                                                                       centroid_row[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=np.RankWarning)
                self.coeffs[i] = self._find_optimal_polynomial(self.rot_row[i][~thrust[i]], self.rot_col[i][~thrust[i]])

            self.polyprime[i] = np.poly1d(self.coeffs[i]).deriv()
            # Compute the arclength s.  It is the length of the polynomial
            # (fitted above) that describes the typical motion.
            x = np.linspace(np.min(self.rot_row[i][~thrust[i]]),
                            np.max(self.rot_row[i][~thrust[i]]), 10000)

            # THIS IS HECKIN' SLOW
            self.s[i] = np.array([self.arclength(x1=np.asarray(xp), x=x, index=i) for xp in self.rot_row[i]])

        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(niters * 5, 4), sharex=True, sharey=True)

        for iter in range(niters):
            # Get rid of the worst quality data.
            mask = np.copy(~thrusters)
            mask |= ~np.isfinite(allflux)
            mask |= np.abs(np.nan_to_num(allflux - np.nanmedian(allflux))) < sigma_1 * np.nanstd(allflux)
            tempflux = self.stitch(alltime, allflux, break_points, mask=mask, knotspacing=self.knotspacing)
            self.bspline = self.fit_bspline(alltime[mask], tempflux[mask], knotspacing=self.knotspacing)

            # Stitch the light curve together and find the best fit smooth.
            _, med, std = sigma_clipped_stats(tempflux, sigma=sigma_1, iters=3)
            mask = np.copy(~thrusters)
            mask |= ~np.isfinite(allflux)
            mask |= np.abs(np.nan_to_num(tempflux/self.bspline(alltime)) - med) < sigma_1 * std
            allflux = self.stitch(alltime, allflux, break_points, mask=mask, knotspacing=self.knotspacing)
            self.bspline = self.fit_bspline(alltime[mask], allflux[mask], knotspacing=self.knotspacing)

            # The SFF algorithm is going to be run on each window independently
            for i in range(self.windows):
                if plot:
                    axs[iter].set_title('Iteration {}'.format(iter + 1))
                    axs[iter].set_xlabel('Arclength [Pixels]')
                    axs[0].set_ylabel('Flux')

                self.trend = np.ones(len(flux[i]))

                iter_trend = self.bspline(time[i])
                self.normflux = flux[i] / iter_trend

                self.trend = iter_trend
                self.interp = self.bin_and_interpolate(self.s[i][~thrust[i]], self.normflux[~thrust[i]],
                                                       sigma=sigma_1)
                # Correct the raw flux
                correction = self.interp(self.s[i])
                if plot:
                    axs[iter].scatter(self.s[i], self.normflux, c='k', alpha=0.4, s=1, zorder=1)
                    axs[iter].scatter(self.s[i][thrust[i]], self.normflux[thrust[i]], c='r', alpha=0.4, s=4, zorder=1)
                    axs[iter].plot(np.sort(self.s[i]), correction[np.argsort(self.s[i])], c='C1', zorder=2)

                corrected_flux = self.normflux / correction
                flux[i] = corrected_flux
                flux[i] *= self.trend

            allflux = np.empty(0)
            for f in flux:
                allflux = np.append(allflux, f)

        if correct_thrusters:
            allflux[thrusters] *= np.nan
        allflux = self.stitch(alltime, allflux, break_points, knotspacing=knotspacing)
        if not restore_trend:
            mask = np.abs(np.nan_to_num(allflux - np.nanmedian(allflux))) < sigma_1 * np.nanstd(allflux)
            mask &= ~thrusters
            bspline = self.fit_bspline(alltime[mask], allflux[mask], knotspacing=self.knotspacing)
            allflux /= bspline(alltime)

        corr = LightCurve(time=alltime, flux=allflux)
        return corr

    def _find_optimal_polynomial(self, x, y, polys=np.arange(2, 6)):
        '''Finds the best fitting polynomial with orders 2 to 6

        Parameters
        ----------
        x, y : np.ndarrays
            Data to fit with a Polynomial
        polys : np.ndarray
            Polynomial orders to fit.

        Returns
        -------
        coeffs : np.ndarray
            Coefficients of the best fit.
        '''
        chis = []
        all_coeffs = []
        for poly in polys:
            all_coeffs.append(np.polyfit(x, y, poly))
            line = np.polyval(all_coeffs[-1], x)
            chis.append(chisquare(y, line, poly - 1).statistic)
        coeffs = all_coeffs[np.argmin(np.asarray(chis))]
        return coeffs

    def rotate_centroids(self, centroid_col, centroid_row):
        """Rotate the coordinate frame of the (col, row) centroids to a new (x,y)
        frame in which the dominant motion of the spacecraft is aligned with
        the x axis.  This makes it easier to fit a characteristic polynomial
        that describes the motion."""
        centroids = np.array([centroid_col, centroid_row])
        _, eig_vecs = linalg.eigh(np.cov(centroids))
        return np.dot(eig_vecs, centroids)

    def _plot_fit(self, i):
        '''Plot the fit of the polynomial to all windows
        '''
        n = int(np.ceil(self.windows**0.5))
        if self.fig is None:
            self.fig, self.axs = plt.subplots(n, n, figsize=(15, 15))
            self.fig.text(0.5, 0.075, 'Rotated Row Centroid', ha='center', fontsize=15)
            self.fig.text(0.075, 0.5, 'Rotated Column Centroid', va='center', rotation='vertical', fontsize=15)

        self.axs[i // n, i % n].scatter(self.rot_row[~self.outliers], self.rot_col[~self.outliers],
                                        c='k', s=1)
        self.axs[i // n, i % n].scatter(self.rot_row[self.outliers], self.rot_col[self.outliers],
                                        c='r', s=3)

        x = np.linspace(min(self.rot_row), max(self.rot_row), 200)
        self.axs[i // n, i % n].plot(x, self.poly(x), '--', lw=3)


    def _plot_rotated_centroids(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.rot_row[~self.outliers], self.rot_col[~self.outliers],
                'ko', markersize=3)
        ax.plot(self.rot_row[~self.outliers], self.rot_col[~self.outliers],
                'bo', markersize=2)
        ax.plot(self.rot_row[self.outliers], self.rot_col[self.outliers],
                'ko', markersize=3)
        ax.plot(self.rot_row[self.outliers], self.rot_col[self.outliers],
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

    def arclength(self, x1, x, index=0):
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
        y = np.sqrt(1 + self.polyprime[index](x[mask]) ** 2)
        return np.trapz(y=y, x=x[mask])

    def fit_bspline(self, time, flux, knotspacing=1.5):
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

    def bin_and_interpolate(self, s, normflux, sigma):

        idx = np.argsort(s)
        s_srtd = s[idx]
        normflux_srtd = normflux[idx]

        self.outlier_mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask
        normflux_srtd = normflux_srtd[(~self.outlier_mask)]
        s_srtd = s_srtd[(~self.outlier_mask)]

        knots = np.array([np.min(s_srtd)]
                         + [np.median(split) for split in np.array_split(s_srtd, self.bins)]
                         + [np.max(s_srtd)])
        bin_means = np.array([normflux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(normflux_srtd, self.bins)]
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
