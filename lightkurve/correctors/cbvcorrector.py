"""Defines KeplerCBVCorrector.
"""
import logging

from tqdm import tqdm

import oktopus
import numpy as np
from matplotlib import pyplot as plt

from astropy.io import fits as pyfits

from .. import MPLSTYLE
from ..lightcurve import KeplerLightCurve
from ..lightcurvefile import KeplerLightCurveFile
from .corrector import Corrector
from ..search import search_cbvs
from .CBVFile import CBVFile

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

    def __init__(self, lc, cbvArray=None, cbvCadenceNo=None, likelihood=oktopus.LaplacianLikelihood,
                 prior=oktopus.LaplacianPrior):
        self.lc = lc
        if not hasattr(self.lc, 'channel'):
            raise ValueError('Input must have a `channel` attribute.')
        self.likelihood = likelihood
        self.prior = prior
        self._ncbvs = CBVFile.nCBVsDefault  # default number of cbvs for Kepler/K2

        if cbvArray is None:
            if self.lc.mission == 'Kepler':
                kCbvFile = search_cbvs(mission=self.lc.mission, quarter=self.lc.quarter)
                cbvs = kCbvFile.get_cbvs(channel=self.lc.channel, cbvIndices='ALL')
            elif self.lc.mission == 'K2':
                kCbvFile = search_cbvs(mission=self.lc.mission, campaign=self.lc.campaign)
                cbvs = kCbvFile.get_cbvs(channel=self.lc.channel, cbvIndices='ALL')
            cbvArray = cbvs.cbvArray
            cbvCadenceNo = cbvs.cbvCadenceNo


        if (cbvArray is not None) & (cbvCadenceNo is None):
            raise ValueError('Please specify both `cbvArray` and `cbvCadenceNo`')

        # Align the CBVs with the lightcurve flux using the cadence numbers
        align_mask = np.in1d(cbvCadenceNo, self.lc.cadenceno)
        self.cbvArray = cbvArray[:,align_mask]
        self.cbvCadenceNo = cbvCadenceNo[align_mask]


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
        clip = np.in1d(np.arange(1, len(self.cbvArray)+1), np.asarray(cbvs))
        time_clip = np.in1d(self.cbvCadenceNo, self.lc.cadenceno)
        def mean_model(*theta):
            coeffs = np.asarray(theta)
            return np.dot(coeffs, self.cbvArray[clip, :][:, time_clip])

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
            clip = np.in1d(np.arange(1, len(self.cbvArray)+1), np.asarray(cbvs))
            time_clip = np.in1d(self.cbvCadenceNo, self.lc.cadenceno)

            if ax is None:
                _, ax = plt.subplots(1)
            for idx, cbv in enumerate(self.cbvArray[clip, :][:, time_clip]):
                ax.plot(self.cbvCadenceNo[time_clip], cbv+idx/10., label='{}'.format(idx + 1))
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
