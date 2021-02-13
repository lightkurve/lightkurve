"""Helper functions for estimating deltanu from periodograms."""
from __future__ import division, print_function

import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy import units as u

from .utils import SeismologyQuantity
from . import utils
from .. import MPLSTYLE

__all__ = ["estimate_deltanu_acf2d", "diagnose_deltanu_acf2d"]


def estimate_deltanu_acf2d(periodogram, numax):
    """Returns the average value of the large frequency spacing, DeltaNu,
    of the seismic oscillations of the target, using an autocorrelation
    function.

    There are many papers on the topic of autocorrelation functions for
    estimating seismic parameters, including but not limited to:
    Roxburgh & Vorontsov (2006), Roxburgh (2009), Mosser & Appourchaux (2009),
    Huber et al. (2009), Verner & Roxburgh (2011) & Viani et al. (2019).

    We base this approach first and foremost off the approach taken in
    Mosser & Appourchaux (2009). Given a known numax, a window around this
    numax is taken of one estimated full-width-half-maximum (FWHM) of the
    seismic mode envelope either side of numax. This width is chosen so that
    the autocorrelation includes all of the visible mode peaks.

    The autocorrelation (numpy.correlate) is given as::

        C = sum(s * s)

    where s is a window of the signal-to-noise spectrum. When shifting
    the spectrum over itself, C will increase when two mode peaks are
    overlapping. Because of the method of this calculation, we need to first
    rescale the power by subtracting its mean, placing its mean around 0. This
    decreases the noise levels in the ACF, as the autocorrelation of the noise
    with itself will be close to zero.

    As is done in Mosser & Appourchaux, we rescale the value of C in terms
    of the noise level in the ACF spectrum as::

        A = (|C^2| / |C[0]^2|) * (2 * len(C) / 3) .

    The method will autocorrelate the region around the estimated numax
    expected to contain seismic oscillation modes. Repeating peaks in the
    autocorrelation implies an evenly spaced structure of modes.
    The peak closest to an empirical estimate of deltanu is taken as the true
    value. The peak finding algorithm is limited by a minimum spacing
    between peaks of 0.5 times the empirical value for deltanu.

    Our empirical estimate for numax is taken from Stello et al. (2009) as::

        deltanu = 0.294 * numax^0.772

    If `numax` is None, a numax is calculated using the estimate_numax()
    function with default settings.

    NOTE: This function is intended for use with solar like Main Sequence
    and Red Giant Branch oscillators only.

    Parameters
    ----------
    numax : float
        An estimated numax value of the mode envelope in the periodogram. If
        not given units it is assumed to be in units of the periodogram
        frequency attribute.

    Returns
    -------
    deltanu : `.SeismologyQuantity`
        The average large frequency spacing of the seismic oscillation modes.
        In units of the periodogram frequency attribute.
    """
    # The frequency grid must be evenly spaced
    if not periodogram._is_evenly_spaced():
        raise ValueError(
            "the ACF 2D method requires that the periodogram "
            "has a grid of uniformly spaced frequencies."
        )

    # Run some checks on the passed in numaxs
    # Ensure input numax is in the correct units
    numax = u.Quantity(numax, periodogram.frequency.unit)
    fs = np.median(np.diff(periodogram.frequency.value))
    if numax.value < fs:
        raise ValueError(
            "The input numax can not be lower than" " a single frequency bin."
        )
    if numax.value > np.nanmax(periodogram.frequency.value):
        raise ValueError(
            "The input numax can not be higher than"
            "the highest frequency value in the periodogram."
        )

    # Calculate deltanu using the method by Stello et al. 2009
    # Make sure that this relation only ever happens in microhertz space
    deltanu_emp = u.Quantity(
        (0.294 * u.Quantity(numax, u.microhertz).value ** 0.772) * u.microhertz,
        periodogram.frequency.unit,
    ).value

    window_width = 2 * int(np.floor(utils.get_fwhm(periodogram, numax.value)))
    aacf = utils.autocorrelate(
        periodogram, numax=numax.value, window_width=window_width
    )
    acf = (np.abs(aacf ** 2) / np.abs(aacf[0] ** 2)) / (3 / (2 * len(aacf)))
    fs = np.median(np.diff(periodogram.frequency.value))
    lags = np.linspace(0.0, len(acf) * fs, len(acf))

    # Select a 25% region region around the empirical deltanu
    sel = (lags > deltanu_emp - 0.25 * deltanu_emp) & (
        lags < deltanu_emp + 0.25 * deltanu_emp
    )

    # Run a peak finder on this region
    peaks, _ = find_peaks(acf[sel], distance=np.floor(deltanu_emp / 2.0 / fs))

    # Select the peak closest to the empirical value
    best_deltanu_value = lags[sel][peaks][
        np.argmin(np.abs(lags[sel][peaks] - deltanu_emp))
    ]
    best_deltanu = u.Quantity(best_deltanu_value, periodogram.frequency.unit)
    diagnostics = {
        "lags": lags,
        "acf": acf,
        "peaks": peaks,
        "sel": sel,
        "numax": numax,
        "deltanu_emp": deltanu_emp,
    }
    result = SeismologyQuantity(
        best_deltanu,
        name="deltanu",
        method="ACF2D",
        diagnostics=diagnostics,
        diagnostics_plot_method=diagnose_deltanu_acf2d,
    )
    return result


def diagnose_deltanu_acf2d(deltanu, periodogram):
    """Returns a diagnostic plot which elucidates how deltanu was estimated.

    [1] Scaled correlation metric vs frequecy lag of the autocorrelation
    window, with inset close up on the determined deltanu and a line
    indicating the determined deltanu.

    For details on the deltanu estimation, see the `estimate_deltanu()`
    function. The calculation performed is identical.

    NOTE: When plotting, we exclude the first two frequency lag bins, to
    make the relevant features on the plot clearer, as these bins are close to
    the spectrum correlated with itself and therefore much higher than the rest
    of the bins.

    Parameters
    ----------
    deltanu : `.SeismologyResult` object
        The object returned by `estimate_deltanu_acf2d()`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The matplotlib axes object.
    """
    with plt.style.context(MPLSTYLE):
        fig, axs = plt.subplots(2, figsize=(8.485, 8))

        ax = axs[0]
        periodogram.plot(ax=ax, label="")
        ax.axvline(
            deltanu.diagnostics["numax"].value, c="r", linewidth=1, alpha=0.4, ls=":"
        )
        ax.text(
            deltanu.diagnostics["numax"].value,
            periodogram.power.value.max() * 0.45,
            "{} ({:.1f} {})".format(
                r"$\nu_{\rm max}$",
                deltanu.diagnostics["numax"].value,
                deltanu.diagnostics["numax"].unit.to_string("latex"),
            ),
            rotation=90,
            ha="right",
            color="r",
            alpha=0.5,
            fontsize=8,
        )
        ax.text(
            0.025,
            0.9,
            "Input Power Spectrum",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=11,
        )

        window_width = 2 * int(
            np.floor(utils.get_fwhm(periodogram, deltanu.diagnostics["numax"].value))
        )
        frequency_spacing = np.median(np.diff(periodogram.frequency.value))
        spread = int(window_width / 2 / frequency_spacing)  # spread in indices

        a = (
            np.argmin(
                np.abs(periodogram.frequency.value - deltanu.diagnostics["numax"].value)
            )
            + spread
        )
        b = (
            np.argmin(
                np.abs(periodogram.frequency.value - deltanu.diagnostics["numax"].value)
            )
            - spread
        )

        a = [
            periodogram.frequency.value[a]
            if a < len(periodogram.frequency)
            else periodogram.frequency.value[-1]
        ][0]
        b = [
            periodogram.frequency.value[b] if b > 0 else periodogram.frequency.value[0]
        ][0]

        ax.axvline(a, c="r", linewidth=2, alpha=0.4, ls="--")
        ax.axvline(b, c="r", linewidth=2, alpha=0.4, ls="--")

        h = periodogram.power.value.max() * 0.9

        ax.annotate(
            "",
            xy=(a, h),
            xytext=(a - (a - b), h),
            arrowprops=dict(arrowstyle="<->", color="r", alpha=0.5),
            va="bottom",
        )
        ax.text(
            a - (a - b) / 2,
            h,
            r"2 $\times$ FWHM",
            color="r",
            alpha=0.7,
            fontsize=10,
            va="bottom",
            ha="center",
        )
        ax.set_xlim(b - ((a - b) * 0.2), a + ((a - b) * 0.2))

        ax = axs[1]
        ax.plot(deltanu.diagnostics["lags"][2:], deltanu.diagnostics["acf"][2:])
        ax.set_xlabel(
            "Frequency Lag [{}]".format(periodogram.frequency.unit.to_string("latex"))
        )
        ax.set_ylabel(r"Scaled Auto Correlation", fontsize=11)

        axin = inset_axes(ax, width="50%", height="50%", loc="upper right")
        axin.set_yticks([])
        axin.plot(
            deltanu.diagnostics["lags"][deltanu.diagnostics["sel"]],
            deltanu.diagnostics["acf"][deltanu.diagnostics["sel"]],
        )
        axin.scatter(
            deltanu.diagnostics["lags"][deltanu.diagnostics["sel"]][
                deltanu.diagnostics["peaks"]
            ],
            deltanu.diagnostics["acf"][deltanu.diagnostics["sel"]][
                deltanu.diagnostics["peaks"]
            ],
            c="r",
            s=5,
        )

        mea_label = r"Measured {} {:.1f} {}".format(
            r"$\Delta\nu$", deltanu.value, periodogram.frequency.unit.to_string("latex")
        )
        ax.axvline(deltanu.value, c="r", linewidth=2, alpha=0.4, label=mea_label)

        emp_label = r"Empirical {} {:.1f} {}".format(
            r"$\Delta\nu$",
            deltanu.diagnostics["deltanu_emp"],
            periodogram.frequency.unit.to_string("latex"),
        )
        ax.axvline(
            deltanu.diagnostics["deltanu_emp"],
            c="b",
            linewidth=2,
            alpha=0.4,
            ls="--",
            label=emp_label,
        )

        axin.axvline(deltanu.value, c="r", linewidth=2, alpha=0.4)
        axin.axvline(
            deltanu.diagnostics["deltanu_emp"], c="b", linewidth=2, alpha=0.4, ls="--"
        )
        ax.text(
            0.025,
            0.9,
            "Scaled Auto Correlation Within 2 FWHM",
            horizontalalignment="left",
            transform=ax.transAxes,
            fontsize=11,
        )

        ax.legend(loc="lower right", fontsize=10)
        return ax
