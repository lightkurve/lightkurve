"""Defines KeplerCBVCorrector.
"""
import logging
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

import oktopus
import numpy as np
from matplotlib import pyplot as plt

from astropy.io import fits as pyfits

from .. import MPLSTYLE
from ..utils import channel_to_module_output
from ..lightcurve import KeplerLightCurve
from ..lightcurvefile import KeplerLightCurveFile
from .corrector import Corrector

log = logging.getLogger(__name__)

__all__ = ['KeplerCBVCorrector']

class KeplerCBVCorrector(Corrector):
    r"""Remove systematic trends from Kepler light curves by fitting
    Cotrending Basis Vectors (CBVs).

    .. math::

        \arg \min_{\bm{\theta} \in \Theta} \sum_{t}|f_{SAP}(t) - \sum_{j=1}^{n}\theta_j v_{j}(t)|^p, p>0, p \in \mathbb{R}

    Attributes
    ----------
    lc : KeplerLightCurveFile, KeplerLightCurve object or str
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

    def __init__(self, lc, cbv_array=None, cbv_cadenceno=None, likelihood=oktopus.LaplacianLikelihood,
                 prior=oktopus.LaplacianPrior):
        self.lc = lc
        if not hasattr(self.lc, 'channel'):
            raise ValueError('Input must have a `channel` attribute.')
        self.likelihood = likelihood
        self.prior = prior
        self._ncbvs = 16  # number of cbvs for Kepler/K2

        if self.lc.mission == 'Kepler':
            self.cbv_base_url = "http://archive.stsci.edu/missions/kepler/cbv/"
        elif self.lc.mission == 'K2':
            self.cbv_base_url = "http://archive.stsci.edu/missions/k2/cbv/"

        if cbv_array is None:
            cbv_array, cbv_cadenceno = self._get_cbv_data(np.arange(1, self._ncbvs))
        if (cbv_array is not None) & (cbv_cadenceno is None):
            raise ValueError('Please specify both `cbv_array` and `cbv_cadenceno`')
        self.cbv_array = cbv_array
        self.cbv_cadenceno = cbv_cadenceno


    @property
    def lc(self):
        return self._lc

    @lc.setter
    def lc(self, value):
        # this enables `lc` to be either a string
        # or an object from KeplerLightCurveFile
        if isinstance(value, str):
            self._lc = KeplerLightCurveFile(value).PDCSAP_FLUX
        elif isinstance(value, KeplerLightCurveFile):
            self._lc = value.SAP_FLUX
        elif isinstance(value, KeplerLightCurve):
            self._lc = value
        else:
            raise ValueError("lc must be either a string, a KeplerLightCurve or a"
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

    def _get_cbv_data(self, cbvs=(1, 2)):
        """Returns the CBV data for a channel and module."""
        module, output = channel_to_module_output(self.lc.channel)
        cbv_file = pyfits.open(self.get_cbv_url())
        cbv_data = cbv_file['MODOUT_{0}_{1}'.format(module, output)].data
        quality_mask = np.in1d(cbv_data['CADENCENO'], self.lc.cadenceno)
        time = cbv_file['MODOUT_{0}_{1}'.format(module, output)].data['CADENCENO'][quality_mask]
        cbv_array = []
        for i in cbvs:
            cbv_array.append(cbv_data.field('VECTOR_{}'.format(i))[quality_mask])
        cbv_array = np.asarray(cbv_array)

        return cbv_array, time

    def correct(self, cbvs=(1, 2), method='powell', options=None):
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
        if options is None:
            options = {}
        median_flux = np.nanmedian(self.lc.flux)
        norm_flux = self.lc.flux / median_flux - 1
        norm_err_flux = self.lc.flux_err / median_flux

        # Trim down to the right number of cbvs
        clip = np.in1d(np.arange(1, len(self.cbv_array)+1), np.asarray(cbvs))
        time_clip = np.in1d(self.cbv_cadenceno, self.lc.cadenceno)
        def mean_model(*theta):
            coeffs = np.asarray(theta)
            return np.dot(coeffs, self.cbv_array[clip, :][:, time_clip])

        prior = self.prior(mean=np.zeros(len(cbvs)), var=16.)
        likelihood = self.likelihood(data=norm_flux, mean=mean_model,
                                     var=norm_err_flux)
        x0 = likelihood.fit(x0=prior.mean, method=method, options=options).x
        posterior = oktopus.Posterior(likelihood=likelihood, prior=prior)

        self._opt_result = posterior.fit(x0=x0, method=method,
                                         options=options)
        self._coeffs = self._opt_result.x
        flux_hat = self.lc.flux - median_flux * mean_model(self._coeffs)
        clc = self.lc.copy()
        clc.flux = flux_hat.reshape(-1)
        return clc

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

        if self.lc.mission == 'Kepler':
            quarter = 'q{:02}'.format(self.lc.quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif self.lc.mission == 'K2':
            campaign = 'c{:02}'.format(self.lc.campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break
        return self.cbv_base_url + cbv_file

    def plot_cbvs(self, cbvs=(1, 2), ax=None):
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
        with plt.style.context(MPLSTYLE):
            clip = np.in1d(np.arange(1, len(self.cbv_array)+1), np.asarray(cbvs))
            time_clip = np.in1d(self.cbv_cadenceno, self.lc.cadenceno)

            if ax is None:
                _, ax = plt.subplots(1)
            for idx, cbv in enumerate(self.cbv_array[clip, :][:, time_clip]):
                ax.plot(self.cbv_cadenceno[time_clip], cbv+idx/10., label='{}'.format(idx + 1))
            ax.set_yticks([])
            ax.set_xlabel('Time (MJD)')
            module, output = channel_to_module_output(self.lc.channel)
            if self.lc.mission == 'Kepler':
                ax.set_title('Kepler CBVs (Module : {}, Output : {}, Quarter : {})'
                             ''.format(module, output, self.lc.quarter))
            elif self.lc.mission == 'K2':
                ax.set_title('K2 CBVs (Module : {}, Output : {}, Campaign : {})'
                             ''.format(module, output, self.lc.campaign))
            ax.grid(':', alpha=0.3)
            ax.legend()
        return ax
