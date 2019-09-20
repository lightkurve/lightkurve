"""Defines RegressionCorrector

Example API
===========
rc = RegressionCorrector(lc, design_matrix=DesignMatrix(...))
lc_corrected = rc.correct()
rc.diagnose()  # returns matplotlib plot
rc.design_matrix.plot()  # returns matplotlib plot
rc.diagnose_weights()  # returns matplotlib plot
rc.coefficients  # numpy array
"""
import numpy as np

from .corrector import Corrector, DesignMatrix, DesignMatrixCollection


class RegressionCorrector(Corrector):
    """Remove noise by regressing against a design matrix of vectors.

    Parameters
    ----------
    lc : `~lightkurve.lightcurve.LightCurve`
        The light curve that needs to be corrected.
    design_matrix : `DesignMatrix` or `DesignMatrixCollection`
        A two-dimensional design matrix with dimensions time x nvectors.
        The vectors contained in this matrix must be known to correlate with
        the noise we want to remove from the light curve.
    """
    def __init__(self, lc, design_matrix):
        if isinstance(design_matrix, DesignMatrix):
            design_matrix = DesignMatrixCollection([design_matrix])
        # Validate user input
        if np.any([~np.isfinite(lc.time), ~np.isfinite(lc.flux), ~np.isfinite(lc.flux_err)]):
            raise ValueError('Input light curve has NaNs in time, flux, and/or flux_err. '
                             'Please remove NaNs before correcting.')
        self.lc = lc
        self.design_matrix = design_matrix


    def _solve_weights(self, cadence_mask=None):
        """Compute the weights w of a given input matrix using np.linalg.solve

        Such that
        model = np.dot(X, w)

        Parameters
        ----------
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        model : np.ndarray
            The best fit model to the data (X dot w)
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux), bool)
        X = self.dm
        flux = self.lc.flux[cadence_mask]
        flux_weights = self.lc.flux_err[cadence_mask]**2

        A = np.dot(X[cadence_mask].T, X[cadence_mask]/flux_weights[:, None])
        B = np.dot(X[cadence_mask].T, flux/flux_weights)
        weights = np.linalg.solve(A, B)
        weights_variance = np.linalg.inv(A)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            weights_sigma = np.diag(weights_variance)**0.5
        return weights, weights_sigma


    def correct(self, cadence_mask=None, preserve_trend=True, sigma=5, niters=5, timescale=3, split=True, **kwargs):
        """Find the best fit correction for the light curve.

        Parameters:
        -----------
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        preserve_trend: bool (default True)
            Whether to preserve or remove long term trends from the returned
            lightcurve
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        timescale : float (default 3)
            Timescale on which to remove long term trends when using the spline corrector.
            Timescale establishes the knotspacing for the spline.
        normalize_and_split: bool (default True)
            Whether to normalize and split the input design matrix. If True, the
            input design matrix will be normalized and split at self.breakindex.
        **kwargs: dict
            Keyword arguments to pass to the optimizer for each `method`.

        Returns:
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Corrected light curve, with noise removed.
        """
        if cadence_mask is None:
            mask = np.ones(len(self.lc.time), bool)
        
        n_knots = int((self.time[-1] - self.time[0])/timescale)
        n_knots = np.max([n_knots, 3])
        for count in range(niters):
            if self.method == 'spline':
                w, var, dm2, model = self._optimize_spline(dm, cadence_mask=mask, n_knots=n_knots)
            if self.method == 'lombscargle':
                w, var, dm2, model = self._optimize_lomb_scargle(dm, cadence_mask=mask, n_knots=n_knots, period=self.period)
            if count != niters - 1:
                mask &= self._clip_outliers(model, sigma=sigma)

        noise = LightCurve(self.time, np.dot(w[:len(dm.T)], dm.T))
        noise.flux -= np.median(noise.flux)
        long_term = LightCurve(self.time, np.dot(w[len(dm.T):], dm2[:, len(dm.T):].T))
        long_term.flux -= np.median(long_term.flux)
        if preserve_trend:
            corrected = self.lc - noise.flux
        else:
            corrected = self.lc - model + np.median(model)
        self.diagnostic_lightcurves = {'noise':noise, 'long_term':long_term, 'corrected':corrected}
        self.cadence_mask = mask
        self.design_matrix = dm2
        return corrected

    def diagnose(self):
        """ Produce diagnostic plots to assess the effectiveness of the correction. """
        ax = self.lc.plot(normalize=False, label='Original', alpha=0.4)
        (self.diagnostic_lightcurves['noise'] + 1).plot(ax=ax, c='b', lw=0.4, label='Noise Model')
        (self.diagnostic_lightcurves['long_term'] + 1).plot(ax=ax, c='r', lw=1, label='Long Term Trend ({})'.format(label))
        ax.axvline(self.time[self.breakindex], ls='--', c='k')

        ax = self.lc.plot(normalize=False, alpha=0.2, label='Original')
        self.lc[~self.cadence_mask].scatter(normalize=False, c='r', marker='x', s=10, label='Outliers', ax=ax)

        self.diagnostic_lightcurves['corrected'].plot(normalize=True, label='Corrected', ax=ax, c='k')
        return


def build_k2_design_matrix(lc):
    """Build a basic design matrix based on the centroid position.

    Builds a design matrix of:

        arclength, arclength**2, arclength**3, darclength/dt
        Column**4, Column**3, Column**2, Column
        Row**4, Row**3, Row**2, Row
        Column**4 Row**3, Column**4 Row **2, Column**4 Row, Column**3 Row**2, Column**2 Row, Column Row
        Row**4 Column**3, Row**4 Column**2, Row**4 Column, Row**3 Column**2, Row**3 Column, Row**2 Column
        Vector Of Ones
    """
    col = lc.centroid_col - lc.centroid_col.min()
    row = lc.centroid_row - lc.centroid_row.min()
    build_components = lambda X, Y: np.array([
                                                ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5,
                                                ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2),
                                                ((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**1.5,
                                                np.gradient(((X - np.min(X) + 1)**2 + (Y - np.min(Y) + 1)**2)**0.5),
                                                X**4, X**3, X**2, X,
                                                Y**4, Y**3, Y**2, Y,
                                                X**4*Y**3, X**4*Y**2, X**4*Y, X**3*Y**2, X**3*Y, X**2*Y, X*Y,
                                                Y**4*X**3, Y**4*X**2, Y**4*X, Y**3*X**2, Y**3*X, Y**2*X]).T

    return build_components(col, row)

    def _optimize_spline(self, design_matrix, cadence_mask=None, n_knots=10):
        """Find the best fitting bspline to the long term trends in the light curve.

        The optimization here is done with a simple linear regression using np.linalg.solve.

        Parameters:
        -----------
        design_matrix : np.ndarray
            Design matrix, with dimensions time x nvectors
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        n_knots : int
            Number of knots to use for the spline. These will be evenly spaced.

        Returns:
        --------
        w : np.ndarray
            Best fit weights of each vector.
        dm : np.ndarray
            The input design matrix contatinated with the lomb-scargle design matrix.
        model : np.ndarray
            The best fit model to the data (X dot w)
        """
        if cadence_mask is None:
            cadence_mask = np.ones(len(lc.flux), bool)
        spline_dm = np.asarray(dmatrix("bs(x, df={}, degree=3, include_intercept=False) - 1".format(n_knots), {"x": self.time}))
        dm = np.hstack([design_matrix, spline_dm, np.atleast_2d(np.ones(len(self.flux))).T])
        w, model, var = self._solve_weights(dm, cadence_mask=cadence_mask)
        return w, var, dm, model
