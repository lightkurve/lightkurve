"""Defines PLDCorrector
"""
from __future__ import division, print_function

import logging
import warnings

import numpy as np

from itertools import combinations_with_replacement as multichoose

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']


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
        noise model for each cadence in time. This function goes up to second
        order, and is represented by

        .. math::

            m_i = \sum_l a_l \frac{f_{il}}{\sum_k f_{ik}} + \sum_l \sum_m b_{lm} \frac{f_{il}f_{im}}{\left( \sum_k f_{ik} \right)^2} + \alpha + \beta t_i + \gamma t_i^2

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
        containing first and second order fractional pixel fluxes.

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
        self.flux = np.nan_to_num(tpf.flux)
        self.flux_err = np.nan_to_num(tpf.flux_err)
        self.time = np.nan_to_num(tpf.time)

    def correct(self, aperture_mask=None, cadence_mask=None, gp_timescale=30,
                n_components_first=None, n_components_second=20, use_gp=True):
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
        n_components_first : int
            Number of first-order PLD components to reduce to with PCA.
            Must be smaller than the number of pixels in the aperture mask.
            If `None`, then 25 or the number of pixels in the mask will be used,
            whichever is smaller.
        n_components_second : int
            Number of second-order PLD components to reduce to with PCA.
        use_gp : boolean
            Option to turn GP fitting on or off.

        Returns
        -------
        corrected_lightcurve : `~lightkurve.lightcurve.LightCurve`
            Returns a corrected lightcurve object. Depending on the input, the
            returned object will be a `KeplerLightCurve`, `TessLightCurve`, or
            general `LightCurve` object.
        """
        # Verify optional dependencies
        try:
            import celerite
        except ImportError:
            log.error("PLD requires the `celerite` Python package. "
                      "See the installation instructions at "
                      "https://docs.lightkurve.org/about/install.html")
            return None
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            log.error("PLD requires the `scikit-learn` Python package. "
                      "See the installation instructions at "
                      "https://docs.lightkurve.org/about/install.html")
            return None

        # Parse the aperture mask to accept strings etc.
        aperture = self.tpf._parse_aperture_mask(aperture_mask)

        # n_components_first cannot be larger than the number of pixels in the mask
        if n_components_first is None:
            n_components_first = min(25, (aperture > 0).sum())

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

        # set default transit mask
        if cadence_mask is None:
            cadence_mask = np.where(self.time)
        M = lambda x: x[cadence_mask]

        # generate flux light curve from desired pixels
        lc = self.tpf.to_lightcurve(aperture_mask=aperture)

        # set aperture values
        aperture_vals = np.copy(aperture_crop).astype(int)

        # `aperture_flux` contains the per-pixel lightcurve in a matrix
        # with shape (n_cadences, n_pixels).
        # We will run PCA on this matrix further below to arrive at the design
        # matrix for the noise model.
        self.aperture_flux = np.array([f*aperture_vals for f in flux_crop]).reshape(len(flux_crop), -1)
        rawflux = np.sum(self.aperture_flux.reshape(len(self.aperture_flux), -1), axis=1)

        # first order PLD
        f1 = self.aperture_flux / rawflux.reshape(-1, 1)
        pca = PCA(n_components=n_components_first)
        X1 = pca.fit_transform(f1)

        # second order PLD
        f2 = np.product(list(multichoose(f1.T, 2)), axis=1).T
        pca = PCA(n_components=n_components_second)
        X2 = pca.fit_transform(f2)

        # Create the design matrix X by stacking X1 and X2 and adding a column
        # vector of 1s for numerical stability (see Luger et al.).
        # X has shape (n_components_first + n_components_second + 1, n_cadences)
        X = np.hstack([np.ones(X1.shape[0]).reshape(-1, 1), X1, X2])

        # mask transits in design matrix
        MX = M(X)

        if use_gp:
            # We use a Gaussian Process to model the long term trend.
            # We do this by estimating the long term trend y by applying the
            # preliminary PLD model defined above and subtracting it from the raw light curve.
            # The "in transit" cadences are masked out in this step to prevent the
            # long term approximation from over-fitting the transits.
            y = M(rawflux) - np.dot(MX, np.linalg.solve(np.dot(MX.T, MX),
                                    np.dot(MX.T, M(rawflux))))
            # Estimate the amplitude parameter of a Matern-3/2 kernel GP
            # by computing the standard deviation of y.
            amp = np.nanstd(y)
            tau = gp_timescale  # tau is a user-defined parameter
            # set up gaussian process using celerite
            # we use a Matern-3/2 kernel for its flexibility and non-periodicity
            kernel = celerite.terms.Matern32Term(np.log(amp), np.log(tau))
            gp = celerite.GP(kernel)

            # recover GP covariance matrix from celerite model
            # sigma is expected to have shape (n_unmasked_cadences, n_unmasked_cadences)
            sigma = gp.get_matrix(M(self.time)) + \
                np.diag(
                    np.sum(M(flux_err_crop).reshape(len(M(flux_err_crop)), -1), axis=1)**2
                       )
        else:
            sigma = np.diag(np.sum(M(flux_err_crop).reshape(len(M(flux_err_crop)),
                            -1), axis=1)**2)

        # compute the coefficients C on the basis vectors;
        # the PLD design matrix will be dotted with C to solve for the noise model.
        A = np.dot(MX.T, np.linalg.solve(sigma, MX))
        B = np.dot(MX.T, np.linalg.solve(sigma, M(rawflux)))
        C = np.linalg.solve(A, B)  # shape (regressors, 1)

        # compute detrended light curve
        model = np.dot(X, C)
        self.detrended_flux = rawflux - (model - np.nanmean(model))

        # Create and return a new LightCurve object with the corrected flux
        corrected_lc = lc.copy()
        corrected_lc.flux = self.detrended_flux
        corrected_lc.flux_err = flux_err
        return corrected_lc
