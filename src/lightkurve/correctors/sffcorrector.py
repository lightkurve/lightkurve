"""Defines the `SFFCorrector` class.

`SFFCorrector` enables systematics to be removed from light curves using the
Self Flat-Fielding (SFF) method described in Vanderburg and Johnson (2014).
"""
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from astropy.units import Quantity

from . import DesignMatrix, DesignMatrixCollection, SparseDesignMatrixCollection
from .regressioncorrector import RegressionCorrector
from .designmatrix import create_spline_matrix, create_sparse_spline_matrix

from .. import MPLSTYLE
from ..utils import LightkurveWarning

log = logging.getLogger(__name__)

__all__ = ["SFFCorrector"]


class SFFCorrector(RegressionCorrector):
    """Special case of `.RegressionCorrector` where the `.DesignMatrix` includes
    the target's centroid positions.

    The design matrix also contains columns representing a spline in time
    design to capture the intrinsic, long-term variability of the target.

    Parameters
    ----------
    lc : `.LightCurve`
        The light curve that needs to be corrected.
    """

    def __init__(self, lc):
        if getattr(lc, "mission", "") == "TESS":
            warnings.warn(
                "The SFF correction method is not suitable for use "
                "with TESS data, because the spacecraft motion does "
                "not proceed along a consistent arc.",
                LightkurveWarning,
            )

        self.raw_lc = lc
        if lc.flux.unit.to_string() == "":
            lc = lc.copy()
        else:
            lc = lc.copy().normalize()

        # Setting these values as None so we don't get a value error if the
        # user tries to access them before "correct()"
        self.window_points = None
        self.windows = None
        self.bins = None
        self.timescale = None
        self.breakindex = None
        self.centroid_col = None
        self.centroid_row = None
        super(SFFCorrector, self).__init__(lc=lc)

    def __repr__(self):
        return "SFFCorrector (LC: {})".format(self.lc.meta.get("TARGETID"))

    def correct(
        self,
        centroid_col=None,
        centroid_row=None,
        windows=20,
        bins=5,
        timescale=1.5,
        breakindex=None,
        degree=3,
        restore_trend=False,
        additional_design_matrix=None,
        polyorder=None,
        sparse=False,
        **kwargs
    ):
        """Find the best fit correction for the light curve.

        Parameters
        ----------
        centroid_col : np.ndarray of floats (optional)
            Array of centroid column positions. If ``None``, will use the
            `centroid_col` attribute of the input light curve by default.
        centroid_row : np.ndarray of floats (optional)
            Array of centroid row positions. If ``None``, will use the
            `centroid_row` attribute of the input light curve by default.
        windows : int
            Number of windows to split the data into to perform the correction.
            Default 20.
        bins : int
            Number of "knots" to place on the arclength spline. More bins will
            increase the number of knots, making the spline smoother in arclength.
            Default 10.
        timescale: float
            Time scale of the b-spline fit to the light curve in time, in units
            of input light curve time.
        breakindex : None, int or list of ints (optional)
            Optionally the user can break the light curve into sections. Set
            break index to either an index at which to break, or list of indicies.
        degree : int
            The degree of polynomials in the splines in time and arclength. Higher
            values will create smoother splines. Default 3.
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        sigma : int (default 5)
            Standard deviation at which to remove outliers from fitting
        niters : int (default 5)
            Number of iterations to fit and remove outliers
        restore_trend : bool (default False)
            Whether to restore the long term spline trend to the light curve
        propagate_errors : bool (default False)
            Whether to propagate the uncertainties from the regression. Default is False.
            Setting to True will increase run time, but will sample from multivariate normal
            distribution of weights.
        additional_design_matrix : `~lightkurve.lightcurve.Correctors.DesignMatrix` (optional)
            Additional design matrix to remove, e.g. containing background vectors.
        polyorder : int
            Deprecated as of Lightkurve v1.4.  Use ``degree`` instead.

        Returns
        -------
        corrected_lc : `~lightkurve.lightcurve.LightCurve`
            Corrected light curve, with noise removed.
        """
        DMC, spline = DesignMatrixCollection, create_spline_matrix
        if sparse:
            DMC, spline = SparseDesignMatrixCollection, create_sparse_spline_matrix

        if polyorder is not None:
            warnings.warn(
                "`polyorder` is deprecated and no longer used, "
                "please use the `degree` keyword instead.",
                LightkurveWarning,
            )

        if centroid_col is None:
            self.lc = self.lc.remove_nans(column="centroid_col")
            centroid_col = self.lc.centroid_col
        if centroid_row is None:
            self.lc = self.lc.remove_nans(column="centroid_row")
            centroid_row = self.lc.centroid_row

        if np.any([~np.isfinite(centroid_row), ~np.isfinite(centroid_col)]):
            raise ValueError("Centroids contain NaN values.")

        self.window_points = _get_window_points(
            centroid_col, centroid_row, windows, breakindex=breakindex
        )
        self.windows = windows
        self.bins = bins
        self.timescale = timescale
        self.breakindex = breakindex
        self.arclength = _estimate_arclength(centroid_col, centroid_row)

        lower_idx = np.asarray(np.append(0, self.window_points), int)
        upper_idx = np.asarray(np.append(self.window_points, len(self.lc.time)), int)

        dms = []
        for idx, a, b in zip(range(len(lower_idx)), lower_idx, upper_idx):
            if isinstance(self.arclength, Quantity):
                ar = np.copy(self.arclength.value)
            else:
                ar = np.copy(self.arclength)

            # Temporary workaround for issue #1161: AstroPy v5.0
            # Masked arrays cannot be passed to `np.in1d` below
            if hasattr(self.arclength, 'mask'):
                ar = ar.unmasked

            knots = list(np.percentile(ar[a:b], np.linspace(0, 100, bins + 1)[1:-1]))
            ar[~np.in1d(ar, ar[a:b])] = 0

            dm = spline(ar, knots=knots, degree=degree).copy()
            dm.columns = [
                "window{}_bin{}".format(idx + 1, jdx + 1) for jdx in range(dm.shape[1])
            ]

            # I'm putting VERY weak priors on the SFF motion vectors
            # (1e-6 is being added to prevent sigma from being zero)
            ps = np.ones(dm.shape[1]) * 10000 * self.lc[a:b].flux.std() + 1e-6
            dm.prior_sigma = ps
            dms.append(dm)

        sff_dm = DMC(dms).to_designmatrix(name="sff")  # .standardize()

        # long term
        n_knots = int((self.lc.time.value[-1] - self.lc.time.value[0]) / timescale)

        s_dm = spline(self.lc.time.value, n_knots=n_knots, name="spline")
        means = [np.average(chunk) for chunk in np.array_split(self.lc.flux, n_knots)]
        #        means = [np.average(self.lc.flux, weights=s_dm.values[:, idx]) for idx in range(s_dm.shape[1])]
        s_dm.prior_mu = np.asarray(means)

        # I'm putting WEAK priors on the spline that it must be around 1
        s_dm.prior_sigma = (
            np.ones(len(s_dm.prior_mu)) * 1000 * self.lc.flux.std().value + 1e-6
        )

        # additional
        if additional_design_matrix is not None:
            if not isinstance(additional_design_matrix, DesignMatrix):
                raise ValueError(
                    "`additional_design_matrix` must be a DesignMatrix object."
                )
            self.additional_design_matrix = additional_design_matrix
            dm = DMC([s_dm, sff_dm, additional_design_matrix])
        else:
            dm = DMC([s_dm, sff_dm])

        # correct
        clc = super(SFFCorrector, self).correct(dm, **kwargs)

        # clean
        if restore_trend:
            trend = self.diagnostic_lightcurves["spline"].flux
            clc += trend - np.nanmedian(trend)
        clc *= self.raw_lc.flux.mean()

        return clc

    def diagnose(self):
        """Returns a diagnostic plot which visualizes what happened during the
        most recent call to `correct()`."""
        axs = self._diagnostic_plot()
        for t in self.window_points:
            axs[0].axvline(self.lc.time.value[t], color="r", ls="--", alpha=0.3)

    def diagnose_arclength(self):
        """Returns a diagnostic plot which visualizes arclength vs flux
        from most recent call to `correct()`."""

        max_plot = 5
        with plt.style.context(MPLSTYLE):
            _, axs = plt.subplots(
                int(np.ceil(self.windows / max_plot)),
                max_plot,
                figsize=(10, int(np.ceil(self.windows / max_plot) * 2)),
                sharex=True,
                sharey=True,
            )
            axs = np.atleast_2d(axs)
            axs[0, 2].set_title("Arclength Plot/Window")
            plt.subplots_adjust(hspace=0, wspace=0)

            lower_idx = np.asarray(np.append(0, self.window_points), int)
            upper_idx = np.asarray(
                np.append(self.window_points, len(self.lc.time)), int
            )
            if hasattr(self, "additional_design_matrix"):
                name = self.additional_design_matrix.name
                f = (
                    self.lc.flux
                    - self.diagnostic_lightcurves["spline"].flux
                    - self.diagnostic_lightcurves[name].flux
                )
            else:
                f = self.lc.flux - self.diagnostic_lightcurves["spline"].flux

            m = self.diagnostic_lightcurves["sff"].flux

            idx, jdx = 0, 0
            for a, b in zip(lower_idx, upper_idx):
                ax = axs[idx, jdx]
                if jdx == 0:
                    ax.set_ylabel("Flux")

                ax.scatter(self.arclength[a:b], f[a:b], s=1, label="Data")
                ax.scatter(
                    self.arclength[a:b][~self.cadence_mask[a:b]],
                    f[a:b][~self.cadence_mask[a:b]],
                    s=10,
                    marker="x",
                    c="r",
                    label="Outliers",
                )

                s = np.argsort(self.arclength[a:b])
                ax.scatter(
                    self.arclength[a:b][s],
                    (m[a:b] - np.median(m[a:b]) + np.median(f[a:b]))[s],
                    c="C2",
                    s=0.5,
                    label="Model",
                )
                jdx += 1
                if jdx >= max_plot:
                    jdx = 0
                    idx += 1
                if b == len(self.lc.time):
                    ax.legend()


######################
#  Helper functions  #
######################


def _get_centroid_dm(col, row, name="centroids"):
    """Returns a `.DesignMatrix` containing (col, row) centroid positions
    and transformations thereof.

    Parameters
    ----------
    col : np.ndarray
        centroid column
    row : np.ndarray
        centroid row
    name : str
        Name to pass to `.DesignMatrix` (default: 'centroids').

    Returns
    -------
    dm: np.ndarray
        Design matrix with shape len(c) x 10
    """
    data = [
        col,
        row,
        col ** 2,
        row ** 2,
        col ** 3,
        row ** 3,
        col * row,
        col ** 2 * row,
        col * row ** 2,
        col ** 2 * row ** 2,
    ]
    names = [
        r"col",
        r"row",
        r"col^2",
        r"row^2",
        r"col^3",
        r"row^3",
        r"col \times row",
        r"col^2 \times row",
        r"col \times row^2",
        r"col^2 \times row^2",
    ]
    df = pd.DataFrame(np.asarray(data).T, columns=names)
    return DesignMatrix(df, name=name)


def _get_thruster_firings(arclength):
    """Find locations where K2 fired thrusters

    Parameters
    ----------
    arc : np.ndarray
        arclength as a function of time

    Returns
    -------
    thrusters: np.ndarray of bools
        True at times where thrusters were fired.
    """
    if isinstance(arclength, Quantity):
        arc = np.copy(arclength.value)
    else:
        arc = np.copy(arclength)
    # Rate of change of rate of change of arclength wrt time
    d2adt2 = np.gradient(np.gradient(arc))
    # Fit a Gaussian, most points lie in a tight region, thruster firings are outliers
    g = models.Gaussian1D(amplitude=100, mean=0, stddev=0.01)
    fitter = fitting.LevMarLSQFitter()
    h = np.histogram(
        d2adt2[np.isfinite(d2adt2)], np.arange(-0.5, 0.5, 0.0001), density=True
    )
    xbins = h[1][1:] - np.median(np.diff(h[1]))
    g = fitter(g, xbins, h[0], weights=h[0] ** 0.5)

    # Depending on the orientation of the roll, it is hard to return
    # the point before the firing or the point after the firing.
    # This makes sure we always return the same value, no matter the roll orientation.
    def _start_and_end(start_or_end):
        """Find points at the start or end of a roll."""
        if start_or_end == "start":
            thrusters = (d2adt2 < (g.stddev * -5)) & np.isfinite(d2adt2)
        if start_or_end == "end":
            thrusters = (d2adt2 > (g.stddev * 5)) & np.isfinite(d2adt2)
        # Pick the best thruster in each cluster
        idx = np.array_split(
            np.arange(len(thrusters)),
            np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0],
        )
        m = np.array_split(
            thrusters, np.where(np.gradient(np.asarray(thrusters, int)) == 0)[0]
        )
        th = []
        for jdx, _ in enumerate(idx):
            if m[jdx].sum() == 0:
                th.append(m[jdx])
            else:
                th.append(
                    (
                        np.abs(np.gradient(arc)[idx[jdx]])
                        == np.abs(np.gradient(arc)[idx[jdx]][m[jdx]]).max()
                    )
                    & m[jdx]
                )
        thrusters = np.hstack(th)
        return thrusters

    # Get the start and end points
    thrusters = np.asarray([_start_and_end("start"), _start_and_end("end")])
    thrusters = thrusters.any(axis=0)

    # Take just the first point.
    thrusters = (np.gradient(np.asarray(thrusters, int)) >= 0) & thrusters
    return thrusters


def _get_window_points(
    centroid_col, centroid_row, windows, arclength=None, breakindex=None
):
    """Returns indices where thrusters are fired.

    Parameters
    ----------
    lc : `.LightCurve` object
        Input light curve
    windows: int
        Number of windows to split the light curve into
    arc: np.ndarray
        Arclength for the roll motion
    breakindex: int
        Cadence where there is a natural break. Windows will be automatically put here.
    """

    if arclength is None:
        arclength = _estimate_arclength(centroid_col, centroid_row)

    # Validate break indices
    if isinstance(breakindex, int):
        breakindexes = [breakindex]
    if breakindex is None:
        breakindexes = []
    elif (breakindex[0] == 0) & (len(breakindex) == 1):
        breakindexes = []
    else:
        breakindexes = breakindex

    if not isinstance(breakindexes, list):
        raise ValueError("`breakindex` must be an int or a list")

    # If the user asks for break indices we should still return them,
    # even if there is only 1 window.
    if windows == 1:
        return breakindexes

    # Find evenly spaced window points
    dt = len(centroid_col) / windows
    lower_idx = np.append(0, breakindexes)
    upper_idx = np.append(breakindexes, len(centroid_col))
    window_points = np.hstack(
        [np.asarray(np.arange(a, b, dt), int) for a, b in zip(lower_idx, upper_idx)]
    )

    # Get thruster firings
    thrusters = _get_thruster_firings(arclength)
    for b in breakindexes:
        thrusters[b] = True
    thrusters = np.where(thrusters)[0]

    # Find the nearest point to each thruster firing, unless it's a user supplied break point
    if len(thrusters) > 0:
        window_points = [
            thrusters[np.argmin(np.abs(thrusters - wp))] + 1
            for wp in window_points
            if wp not in breakindexes
        ]
    window_points = np.unique(np.hstack([window_points, breakindexes]))

    # If the first or last windows are very short (<40% median window length),
    # then we add them to the second or penultimate window, respectively,
    # by removing their break points.
    median_length = np.median(np.diff(window_points))
    if window_points[0] < 0.4 * median_length:
        window_points = window_points[1:]
    if window_points[-1] > (len(centroid_col) - 0.4 * median_length):
        window_points = window_points[:-1]

    return np.asarray(window_points, dtype=int)


def _estimate_arclength(centroid_col, centroid_row):
    """Estimate the arclength given column and row centroid positions.

    We use the approximation that the arclength equals

        (row**2 + col**2)**0.5

    For this to work, row and column must be correlated not anticorrelated.
    """
    col = centroid_col - np.nanmin(centroid_col)
    row = centroid_row - np.nanmin(centroid_row)
    if np.all((col == 0) & (row == 0)):
        raise RuntimeError("Arclength cannot be computed because there is no "
                           "centroid motion. Make sure that the aperture of "
                           "the TPF at least two pixels.")
    # Force c to be correlated not anticorrelated
    if np.polyfit(col.data, row.data, 1)[0] < 0:
        col = np.nanmax(col) - col
    arclength = (col ** 2 + row ** 2) ** 0.5
    return arclength 
