"""Defines KeplerCBVCorrector and SFFCorrector."""

from __future__ import division, print_function

import logging
import requests
import warnings

from bs4 import BeautifulSoup
from tqdm import tqdm

import oktopus
import numpy as np
from scipy import linalg, interpolate
from matplotlib import pyplot as plt

from astropy.io import fits as pyfits
from astropy.stats import sigma_clip

from scipy.stats import chisquare

from .utils import channel_to_module_output
from .lightcurve import LightCurve
from .lightcurvefile import KeplerLightCurveFile

log = logging.getLogger(__name__)

__all__ = ['KeplerCBVCorrector', 'SFFCorrector']


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
        time = cbv_file['MODOUT_{0}_{1}'.format(module, output)].data['TIME_MJD'][self.lc_file.quality_mask]
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
            ax.set_title('Kepler CBVs (Module : {}, Output : {}, Quarter : {})'
                         ''.format(module, output, self.lc_file.quarter))
        elif self.lc_file.mission == 'K2':
            ax.set_title('K2 CBVs (Module : {}, Output : {}, Campaign : {})'
                         ''.format(module, output, self.lc_file.campaign))
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


    def christina_correct(self, origtime, origflux, origcentroid_col, origcentroid_row, bins=15, windows=1,
                            niters=3, sigma_1=3., knots=1.5, window_shift=0,
                            sigma_2=5., restore_trend=False, plot=False):
        self.windows = windows
        self.bins = bins
        self.knots = knots
        self.nw = int(np.ceil(self.windows**0.5))

        alltime = np.copy(origtime)
        allflux = np.copy(origflux)

        # How many points are there per window>
        dw = len(origtime) // self.windows

        #We shouldn't be able to shift the window more than this...
        window_shift = window_shift % dw

        if window_shift != 0:
            self.windows+=1
            
        # Split up time, flux, column and row into windows
        time = np.split(origtime, np.arange(self.windows) * dw + window_shift)
        flux = np.split(origflux, np.arange(self.windows) * dw + window_shift)
        centroid_col = np.split(origcentroid_col, np.arange(self.windows) * dw + window_shift)
        centroid_row = np.split(origcentroid_row, np.arange(self.windows) * dw + window_shift)

        if window_shift == 0:
            time = time[1:]
            flux = flux[1:]
            centroid_col = centroid_col[1:]
            centroid_row = centroid_row[1:]
#        import pdb;pdb.set_trace()

        for iter in range(niters):
            mask = np.abs(allflux - np.median(allflux)) < sigma_1 * allflux.std()
            self.bspline = self.fit_bspline(alltime[mask], allflux[mask], knot_separation=knots)
#            plt.plot(alltime, allflux)
#            plt.plot(alltime, self.bspline(alltime))
#            plt.show()
#            import pdb; pdb.set_trace();
            # The SFF algorithm is going to be run on each window independently
            if plot:
                fig = plt.figure(figsize=(15, 2*self.windows))
                plt.suptitle('Iteration {}'.format(iter + 1), fontsize=20)
            for i in range(self.windows):
                # To make it easier (and more numerically stable) to fit a
                # characteristic polynomial that describes the spacecraft motion,
                # we rotate the centroids to a new coordinate frame in which
                # the dominant direction of motion is aligned with the x-axis.
                self.rot_col, self.rot_row = self.rotate_centroids(centroid_col[i],
                                                                   centroid_row[i])
                # Next, we fit the motion polynomial after removing outliers
                self.outliers = sigma_clip(data=self.rot_col,
                                               sigma=sigma_2).mask
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=np.RankWarning)
                    coeffs = self._find_optimal_polynomial(self.rot_row[~self.outliers], self.rot_col[~self.outliers])

                self.poly = np.poly1d(coeffs)
                self.polyprime = np.poly1d(coeffs).deriv()
#                if plot:
#                    self._plot_fit(i)

                # Compute the arclength s.  It is the length of the polynomial
                # (fitted above) that describes the typical motion.
                x = np.linspace(np.min(self.rot_row[~self.outliers]),
                                np.max(self.rot_row[~self.outliers]), 10000)
                self.s = np.array([self.arclength(x1=xp, x=x) for xp in self.rot_row])
                self.trend = np.ones(len(flux[i]))

                iter_trend = self.bspline(time[i])
                self.normflux = flux[i] / iter_trend
                self.trend = iter_trend

                if plot:
                    ax = plt.subplot2grid((self.windows, 4), (i, 0), colspan=3)
                    ax.plot(time[i], flux[i], color='k')
                    if remove_trend:
                        ax.plot(time[i], self.bspline(time[i]), color='r')
                    ax = plt.subplot2grid((self.windows, 4), (i, 3), colspan=1)
                    ax.scatter(self.s, flux[i]/np.median(flux[i]), c='k', s=1, label='Flux')
                    if remove_trend:
                        ax.scatter(self.s, self.normflux, c='r', s=1, label='Normed Flux')
                    ax.legend(fontsize=8)

                self.interp = self.bin_and_interpolate(self.s, self.normflux,
                                                       sigma=sigma_1)
                # Correct the raw flux
                correction = self.interp(self.s)
#                correction /= (np.median(correction/self.normflux))
#                correction /= np.median(correction)
                corrected_flux = self.normflux / correction
#                plt.figure()
#                plt.plot(self.normflux*self.trend)
#                plt.plot(flux[i])
#                plt.plot(corrected_flux)
                flux[i] = corrected_flux
                if restore_trend:
                    flux[i] *= self.trend
            allflux = np.empty(0)
            for f in flux:
                allflux = np.append(allflux, f)


        corr = LightCurve(time=alltime, flux=allflux)
#        corr.plot(c='k')
#        for t in time:
#            plt.axvline(t[0], ls='--')
        return corr

    def correct(self, time, flux, centroid_col, centroid_row, bins=15, windows=1,
                polyorder=5, niters=3, sigma_1=3.,
                sigma_2=5., restore_trend=False, plot=False):
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
        self.windows = windows
        self.bins = bins
        # Split up time, flux, column and row into windows
        self.bspline = self.fit_bspline(time, flux)

        time = np.array_split(time, self.windows)
        flux = np.array_split(flux, self.windows)
        centroid_col = np.array_split(centroid_col, self.windows)
        centroid_row = np.array_split(centroid_row, self.windows)

        # If there are too many windows, the arrays get too small, and SFF won't work.
        if len(time[0]) <= 3:
            raise SFFError('Window size is too high, too few points in window.')
        if len(time[0]) < 20:
            log.warning('Window size is too high, less than 20 points per window.')

        flux_hat = np.array([])
        nw = int(np.ceil(self.windows**0.5))
#        fig, axs = plt.subplots(nw, nw, figsize=(15, 15))

        # The SFF algorithm is going to be run on each window independently
        for i in range(self.windows):
            # To make it easier (and more numerically stable) to fit a
            # characteristic polynomial that describes the spacecraft motion,
            # we rotate the centroids to a new coordinate frame in which
            # the dominant direction of motion is aligned with the x-axis.
            self.rot_col, self.rot_row = self.rotate_centroids(centroid_col[i],
                                                               centroid_row[i])
            # Next, we fit the motion polynomial after removing outliers
            self.outliers = sigma_clip(data=self.rot_col,
                                           sigma=sigma_2).mask
            with warnings.catch_warnings():
                # ignore warning messages related to polyfit being poorly conditioned
                warnings.simplefilter("ignore", category=np.RankWarning)
                if polyorder is None:
                    # Find the best polynomial order to fit through chisquare minimization
                    coeffs = self._find_optimal_polynomial(self.rot_row[~self.outliers], self.rot_col[~self.outliers])
                else:
                    coeffs = np.polyfit(self.rot_row[~self.outliers],
                                        self.rot_col[~self.outliers], polyorder)

            self.poly = np.poly1d(coeffs)
            self.polyprime = np.poly1d(coeffs).deriv()
            if plot:
                self._plot_fit(i)

            # Compute the arclength s.  It is the length of the polynomial
            # (fitted above) that describes the typical motion.
            x = np.linspace(np.min(self.rot_row[~self.outliers]),
                            np.max(self.rot_row[~self.outliers]), 10000)
            self.s = np.array([self.arclength(x1=xp, x=x) for xp in self.rot_row])

            # Next, we find and apply the correction iteratively
            self.trend *= np.median(flux[i])
            self.normflux = flux[i]/np.median(flux[i])


            fig = plt.figure(figsize=(15,3))
            for n in range(niters):
                # Measure the correlation between arclength and flux
                coeff = self._find_optimal_polynomial(self.s, flux[i] / np.median(flux[i]))
                residuals = flux[i] / np.median(flux[i]) - np.polyval(coeff, self.s)

                # Remove the long-term variation by dividing the flux by the spline
                iter_trend = self.bspline(time[i])

                # Measure the correlation between arclength and NORMED flux
                normed_coeff = self._find_optimal_polynomial(self.s, flux[i] / iter_trend)
                normed_residuals = flux[i] / iter_trend - np.polyval(normed_coeff, self.s)

                # For the first iteration, don't remove trends.
                if n == 0:
                    normed_residuals *= 1e10

                # If the normalization improved the correlation, then use it...
                if normed_residuals.std() < residuals.std():
                    self.normflux = flux[i] / iter_trend
                    self.trend *= iter_trend
                    print('normalizing')
                # Otherwise, don't do anything.
                else:
                    self.trend *= np.median(flux[i])
                    self.normflux = flux[i]/np.median(flux[i])
                    print('leaving')
                # Bin and interpolate normalized flux to capture the dependency
                # of the flux as a function of arclength
                ax = plt.subplot2grid((1, 4), (0, 0), colspan=3)
                ax.plot(time[i], flux[i])
                ax.plot(time[i], self.bspline(time[i]))
                ax = plt.subplot2grid((1, 4), (0, 3), colspan=1)
                ax.scatter(self.s, flux[i]/np.median(flux[i]), c='k', s=1, label='Unnormalized')
                ax.scatter(self.s, np.polyval(coeff, self.s), c='r', s=1, label='Unnormalized')

                ax.scatter(self.s, self.normflux, c='C{}'.format(n), s=1, label='Normalized')
                ax.scatter(self.s, np.polyval(normed_coeff, self.s), c='purple', s=1, label='Unnormalized')

                ax.legend(fontsize=8)


#                plt.show()
#                import pdb;pdb.set_trace()
#                axs[i // nw, i % nw].scatter(self.s, flux[i],
#                                                c='r'.format(n), s=1)

#                axs[i // nw, i % nw].scatter(self.s, self.normflux,
#                                                c='C{}'.format(n), s=1)
                self.interp = self.bin_and_interpolate(self.s, self.normflux,
                                                       sigma=sigma_1)
                # Correct the raw flux
                corrected_flux = self.normflux / self.interp(self.s)
                flux[i] = corrected_flux

            if restore_trend:
                flux[i] *= self.trend
            flux_hat = np.append(flux_hat, flux[i])

        corrected =  LightCurve(time=timecopy, flux=flux_hat)
        return corrected

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

    def fit_bspline(self, time, flux, s=0, knot_separation=1.5):
        """s describes the "smoothness" of the spline"""
        knots = np.arange(time[0] + knot_separation, time[-1] - knot_separation, knot_separation)
        t, c, k = interpolate.splrep(time, flux, t=knots[1:], s=s, task=-1)
        return interpolate.BSpline(t, c, k)

    def bin_and_interpolate(self, s, normflux, sigma):
        idx = np.argsort(s)
        s_srtd = s[idx]
        normflux_srtd = normflux[idx]

        self.outlier_mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask
        normflux_srtd = normflux_srtd[~self.outlier_mask]
        s_srtd = s_srtd[~self.outlier_mask]

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
