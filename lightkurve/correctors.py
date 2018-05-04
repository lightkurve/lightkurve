"""Defines KeplerCBVCorrector and SFFCorrector."""

from __future__ import division, print_function

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
from astropy.stats import sigma_clip

from .utils import channel_to_module_output
from .lightcurve import LightCurve
from .lightcurvefile import KeplerLightCurveFile

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

    def __init__(self, lc, mask=None):
        if np.any([np.isnan(lc.flux), np.isnan(lc.time), np.isnan(lc.flux_err)]):
            raise ValueError('Light curve contains NaN values.')
        if np.nansum(lc.normalize().flux - lc.flux) != 0:
            raise ValueError('Flux is unnormalized.')
        self.lc = lc
        self.break_tolerance = 10
        if mask is None:
            self.mask = np.ones(len(lc.flux), dtype=bool)
        else:
            self.mask = mask
        self.dt = self.lc.time[1:] - self.lc.time[0:-1]
        self.dt_min = np.nanmin(self.dt)
        self.breaks = self.find_breaks()

    def initialize(self, timescale=None):
        '''Returns a corrected lightcurve which is a copy of lc.

        Parameters
        ----------
        timescale : float or None
            If not None, sets the lower bound for optimization of the GP. Will
            not fit trends on timescales near to or shorter than this timescale.
        '''
        # Set up the kernels and bounds for celerite
        if timescale is not None:
            bounds = [(-15, 15), (np.log(5 * timescale), 15)]
            log_rho = np.log(5 * timescale)
        else:
            bounds = [(-15, 15), (-15, 15)]
            log_rho = -np.log(0.26)
        log_sigma = np.log(np.nanstd(self.lc[self.mask].flux))
        jittersigma = np.log(np.nanmedian(self.lc.flux_err))
        kernel = celerite.terms.Matern32Term(log_sigma=log_sigma,
                                             log_rho=log_rho, bounds=bounds)
        kernel += celerite.terms.JitterTerm(log_sigma=jittersigma)
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
                        method="L-BFGS-B", bounds=bounds, args=(self.lc[self.mask].flux, gp))
        gp.set_parameter_vector(soln.x)
        self.best_fit_params = gp.get_parameter_dict()
        self.gp = gp

    def correct(self, timescale=0.25, iters=2, sigma=3, return_trend=False):
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
        boxwidth = (int(timescale//self.dt_min))
        if boxwidth < 1:
            raise MaskError('timescale is too short ({})'.format(timescale))
        mask = np.ones(len(self.lc.flux), dtype=bool)
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
            self.initialize(timescale=timescale)
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
            mask = (convolve(mask, Box1DKernel(boxwidth)) == 1)

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
        for n in tqdm(range(2, self._ncbvs+1)):
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
            Matplotlib axis object. If none, one will be generated.
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
        pass

    def correct(self, time, flux, centroid_col, centroid_row,
                polyorder=5, niters=3, bins=15, windows=1, sigma_1=3.,
                sigma_2=5., restore_trend=False):
        """Returns a systematics-corrected LightCurve.

        Note that it is assumed that time and flux do not contain NaNs.

        Parameters
        ----------
        time : array-like
            Time measurements
        flux : array-like
            Data flux for every time point
        centroid_col, centroid_row : array-like, array-like
            Centroid column and row coordinates as a function of time
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
        corrected_lightcurve : LightCurve object
            Returns a corrected lightcurve object.
        """
        timecopy = time
        time = np.array_split(time, windows)
        flux = np.array_split(flux, windows)
        centroid_col = np.array_split(centroid_col, windows)
        centroid_row = np.array_split(centroid_row, windows)

        flux_hat = np.array([])
        # The SFF algorithm is going to be run on each window independently

        for i in tqdm(range(windows)):
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

            # Next, we find and apply the correction iteratively
            self.trend = np.ones(len(time[i]))
            for n in range(niters):
                # First, fit a spline to capture the long-term varation
                # We don't want to fit the long-term trend because we know
                # that the K2 motion noise is a high-frequency effect.
                self.bspline = self.fit_bspline(time[i], flux[i])
                # Remove the long-term variation by dividing the flux by the spline
                iter_trend = self.bspline(time[i] - time[i][0])
                self.normflux = flux[i] / iter_trend
                self.trend *= iter_trend
                # Bin and interpolate normalized flux to capture the dependency
                # of the flux as a function of arclength
                self.interp = self.bin_and_interpolate(self.s, self.normflux, bins,
                                                       sigma=sigma_1)
                # Correct the raw flux
                corrected_flux = self.normflux / self.interp(self.s)
                flux[i] = corrected_flux

            if restore_trend:
                flux[i] *= self.trend
            flux_hat = np.append(flux_hat, flux[i])

        return LightCurve(time=timecopy, flux=flux_hat)

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

    def fit_bspline(self, time, flux, s=0):
        """s describes the "smoothness" of the spline"""
        time = time - time[0]
        knots = np.arange(0, time[-1], 1.5)
        t, c, k = interpolate.splrep(time, flux, t=knots[1:], s=s, task=-1)
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
