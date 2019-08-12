"""Defines PLDCorrector
"""
from __future__ import division, print_function

import logging
import warnings
from itertools import combinations_with_replacement as multichoose

import numpy as np

from .corrector import Corrector

log = logging.getLogger(__name__)

__all__ = ['PLDCorrector']


class PLDCorrector(Corrector):
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
                use_gp=True, pld_order=2, n_pca_terms=10, pld_aperture_mask=None):
        r"""Returns a PLD systematics-corrected LightCurve.

        Parameters
        ----------
        aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used to generate the raw flux light curve.
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
        pld_aperture_mask : array-like, 'pipeline', 'all', 'threshold', or None
            A boolean array describing the aperture such that `True` means
            that the pixel will be used when selecting the PLD basis vectors.
            If `None` or `all` are passed in, all pixels will be used.
            If 'pipeline' is passed, the mask suggested by the official pipeline
            will be returned.
            If 'threshold' is passed, all pixels brighter than 3-sigma above
            the median flux will be used.

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

        # generate flux light curve from desired pixels
        lc = self.tpf.to_lightcurve(aperture_mask=aperture)
        rawflux = lc.flux
        rawflux_err = lc.flux_err

        # create nan mask
        nanmask = np.isfinite(self.time)
        nanmask &= np.isfinite(rawflux)
        nanmask &= np.isfinite(rawflux_err)
        nanmask &= np.abs(rawflux_err) > 1e-12

        # mask out nan values
        rawflux = rawflux[nanmask]
        rawflux_err = rawflux_err[nanmask]
        self.flux = self.flux[nanmask]
        self.flux_err = self.flux_err[nanmask]
        self.time = self.time[nanmask]

        # parse the PLD aperture mask
        pld_pixel_mask = self.tpf._parse_aperture_mask(pld_aperture_mask)

        # find pixel bounds of aperture on tpf
        xmin, xmax = min(np.where(pld_pixel_mask)[0]),  max(np.where(pld_pixel_mask)[0])
        ymin, ymax = min(np.where(pld_pixel_mask)[1]),  max(np.where(pld_pixel_mask)[1])

        # crop data cube to include only desired pixels
        # this is required for superstamps to ensure matrix is invertable
        flux_crop = self.flux[:, xmin:xmax+1, ymin:ymax+1]
        flux_err_crop = self.flux_err[:, xmin:xmax+1, ymin:ymax+1]
        aperture_crop = pld_pixel_mask[xmin:xmax+1, ymin:ymax+1]

        # calculate errors (ignore warnings related to zero or negative errors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            flux_err = np.nansum(flux_err_crop[:, aperture_crop]**2, axis=1)**0.5

        # first order PLD design matrix
        pld_flux = flux_crop[:, aperture_crop]
        f1 = np.reshape(pld_flux, (len(pld_flux), -1))
        X1 = f1 / np.nansum(pld_flux, axis=-1)[:, None]
        # No NaN pixels
        X1 = X1[:, np.isfinite(X1).all(axis=0)]

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
            cadence_mask = np.ones_like(lc.time, dtype=bool)
        M = lambda x: x[cadence_mask[nanmask]]

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
        corrected_lc = lc.copy()[nanmask]
        corrected_lc.flux = self.detrended_flux
        corrected_lc.flux_err = flux_err
        return corrected_lc
