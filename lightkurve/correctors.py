"""Defines KeplerCBVCorrector, SFFCorrector, and PLDCorrector.

These classes are intended to help remove instrument systematics from
time series photometry data.
"""
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

from .utils import channel_to_module_output
from .lightcurve import LightCurve
from .lightcurvefile import KeplerLightCurveFile

from itertools import combinations_with_replacement as multichoose

log = logging.getLogger(__name__)

__all__ = ['SFFCorrector', 'PLDCorrector', 'KeplerCBVCorrector']


class KeplerCBVCorrector(object):
    r"""Remove systematic trends from Kepler light curves by fitting
    Cotrending Basis Vectors (CBVs).

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


class PLDCorrector(object):
    r"""Implements the Pixel Level Decorrelation (PLD) systematics removal method.

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

            m_i = \alpha + \beta t_i + \gamma t_i^2 + \sum_l a_l \frac{f_{il}}{\sum_k f_{ik}} + \sum_l \sum_m b_{lm} \frac{f_{il}f_{im}}{\left( \sum_k f_{ik} \right)^2} + ...
        where

          - :math:`m_i` is the noise model at time :math:`t_i`
          - :math:`f_{il}` is the flux in the :math:`l^\text{th}` pixel at time :math:`t_i`
          - :math:`a_l` is the first-order PLD coefficient on the linear term
          - :math:`b_{lm}` is the second-order PLD coefficient on the :math:`l^\text{th}`,
            :math:`m^\text{th}` pixel pair
          - :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are the
            Gaussian Process terms applied to capture long-period variability.

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
    def __init__(self, tpf):
        self.tpf = tpf

        self.flux = tpf.flux
        self.flux_err = tpf.flux_err
        self.time = tpf.time

    def correct(self, aperture_mask=None, cadence_mask=None, gp_timescale=30,
                use_gp=True, pld_order=2, n_pca_terms=10):
        r"""Returns a PLD systematics-corrected LightCurve.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used.
            If `None` or 'all' are passed, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.
        cadence_mask : array-like
            A mask that will be applied to the cadences prior to constructing
            the detrending model. For example, you can pass a boolean array
            of length `n_cadences` where `True` means that the cadence will be
            included in the noise model. You may also pass an array of indices.
            This option enables signals of interest (e.g. planet transits)
            to be excluded from the noise model, which will prevent over-fitting.
            By default, no cadences will be masked.
        gp_timescale : float
            Gaussian Process time scale length term (`tau`) used to define
            length of fit variability in days.
        use_gp : boolean
            Option to turn GP fitting on or off.  You would typically only set
            this to False to speed up the correction (at the cost of precision),
            or if you suspect the presence of systematic noise at long timescales.
        pld_order : int
            The order of Pixel Level De-correlation to be performed. First order
            (`n=1`) uses only the pixel fluxes to construct the design matrix.
            Higher order populates the design matrix with columns constructed
            from the products of pixel fluxes.
        n_pca_terms : int
            Number of terms added to the design matrix from each order of PLD
            when performing Principal Component Analysis for models higher than
            first order. Increasing this value may provide higher precision at
            the expense of computational time.

        Returns
        -------
        corrected_lightcurve : `~lightkurve.lightcurve.LightCurve`
            Returns a corrected lightcurve object. Depending on the input, the
            returned object will be a `KeplerLightCurve`, `TessLightCurve`, or
            general `LightCurve` object.
        """
        if use_gp:
            # Verify optional dependency
            try:
                import celerite
            except ImportError:
                log.error("PLD uses the `celerite` Python package. "
                          "See the installation instructions at "
                          "https://docs.lightkurve.org/about/install.html. "
                          "`use_gp` has been set to `False`.")
                use_gp = False

        # Parse the aperture mask to accept strings etc.
        aperture = self.tpf._parse_aperture_mask(aperture_mask)

        # find pixel bounds of aperture on tpf
        xmin, xmax = min(np.where(aperture)[0]),  max(np.where(aperture)[0])
        ymin, ymax = min(np.where(aperture)[1]),  max(np.where(aperture)[1])

        # crop data cube to include only desired pixels
        # this is required for superstamps to ensure matrix is invertable
        flux_crop = self.flux[:, xmin:xmax+1, ymin:ymax+1]
        flux_err_crop = self.flux_err[:, xmin:xmax+1, ymin:ymax+1]
        aperture_crop = aperture[xmin:xmax+1, ymin:ymax+1]

        # calculate errors (ignore warnings related to zero or negative errors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = np.nansum(flux_err_crop[:, aperture_crop]**2, axis=1)**0.5

        # generate flux light curve from desired pixels
        lc = self.tpf.to_lightcurve(aperture_mask=aperture)
        rawflux = lc.flux
        rawflux_err = lc.flux_err

        # first order PLD design matrix
        pld_flux = flux_crop[:, aperture_crop]
        f1 = np.reshape(pld_flux, (len(pld_flux), -1))
        X1 = f1 / np.sum(pld_flux, axis=-1)[:, None]

        # higher order PLD design matrices
        X_sections = [np.ones((len(flux_crop), 1)), X1]
        for i in range(2, pld_order+1):
            f2 = np.product(list(multichoose(X1.T, pld_order)), axis=1).T
            try:
                # We use an optional dependency for very fast PCA (fbpca).
                # If the import fails we will fall back on using the slower `np.linalg.svd`
                from fbpca import pca
                components, _, _ = pca(f2, n_pca_terms)
            except ImportError:
                log.error("PLD uses the `fbpca` package. You can pip install "
                          "with `pip install fbpca`. Using `np.linalg.svd` "
                          "instead.")
                components, _, _ = np.linalg.svd(f2)
            X_n = components[:, :n_pca_terms]
            X_sections.append(X_n)

        # Create the design matrix X by stacking X1 and higher order components, and
        # adding a column vector of 1s for numerical stability (see Luger et al.).
        # X has shape (n_components_first + n_components_higher_order + 1, n_cadences)
        X = np.concatenate(X_sections, axis=1)

        # set default transit mask
        if cadence_mask is None:
            cadence_mask = np.ones_like(self.time, dtype=bool)
        m = np.zeros_like(self.time, dtype=bool)
        m[cadence_mask] = True

        # mask out any infinite or nan indices
        m &= np.isfinite(self.time)
        m &= np.isfinite(rawflux)
        m &= np.isfinite(rawflux_err)
        m &= np.abs(rawflux_err) > 1e-12

        # create mask function
        cadence_mask = m
        M = lambda x: x[cadence_mask]

        # mask transits in design matrix
        MX = M(X)

        if use_gp:
            # We use a Gaussian Process to model the long term trend.
            # We do this by estimating the long term trend y by applying the
            # preliminary PLD model defined above and subtracting it from the raw light curve.
            # The "in transit" cadences are masked out in this step to prevent the
            # long term approximation from over-fitting the transits.
            XTX = np.dot(MX.T, MX)
            XTX[np.diag_indices_from(XTX)] += 1e-8
            XTy = np.dot(MX.T, M(rawflux))
            y = M(rawflux) - np.dot(MX, np.linalg.solve(XTX, XTy))

            # Estimate the amplitude parameter of a Matern-3/2 kernel GP
            # by computing the standard deviation of y.
            amp = np.nanstd(y)
            tau = gp_timescale  # tau is a user-defined parameter
            # set up gaussian process using celerite
            # we use a Matern-3/2 kernel for its flexibility and non-periodicity
            kernel = celerite.terms.Matern32Term(np.log(amp), np.log(tau))
            gp = celerite.GP(kernel)
            gp.compute(M(self.time), M(rawflux_err))

            # compute the coefficients C on the basis vectors;
            # the PLD design matrix will be dotted with C to solve for the noise model.
            A = np.dot(MX.T, gp.apply_inverse(MX))
            B = np.dot(MX.T, gp.apply_inverse(M(rawflux)[:, None])[:, 0])

        else:
            # compute the coefficients C on the basis vectors;
            # the PLD design matrix will be dotted with C to solve for the noise model.
            ivar = 1.0 / M(rawflux_err)**2 # inverse variance
            A = np.dot(MX.T, MX * ivar[:, None])
            B = np.dot(MX.T, M(rawflux) * ivar)

        # apply prior to design matrix weights for numerical stability
        A[np.diag_indices_from(A)] += 1e-8
        C = np.linalg.solve(A, B)

        # compute detrended light curve
        model = np.dot(X, C)
        self.detrended_flux = rawflux - (model - np.nanmean(model))

        # Create and return a new LightCurve object with the corrected flux
        corrected_lc = lc.copy()
        corrected_lc.flux = self.detrended_flux
        corrected_lc.flux_err = flux_err
        return corrected_lc
