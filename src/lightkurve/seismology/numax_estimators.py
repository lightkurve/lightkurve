"""Helper functions for estimating numax from periodograms."""
import numpy as np
from matplotlib import pyplot as plt

from astropy.convolution import convolve, Gaussian1DKernel
from astropy import units as u

from .. import MPLSTYLE
from . import utils
from .utils import SeismologyQuantity

__all__ = ["estimate_numax_acf2d", "diagnose_numax_acf2d"]


def estimate_numax_acf2d(periodogram, numaxs=None, window_width=None, spacing=None):
    """Estimates the peak of the envelope of seismic oscillation modes, numax,
    using an autocorrelation function.

    There are many papers on the topic of autocorrelation functions for
    estimating seismic parameters, including but not limited to:
    Roxburgh & Vorontsov (2006), Roxburgh (2009), Mosser & Appourchaux (2009),
    Huber et al. (2009), Verner & Roxburgh (2011) & Viani et al. (2019).

    We base this approach first and foremost off the 2D ACF numax estimation
    presented in Viani et al. (2019) and other papers above. A window of
    fixed width (either given by the user, 25 microhertz for Red Giants or
    250 microhertz for Main Sequence stars) is moved along the power
    spectrum, where the central frequency of the window moves in steps of 1
    microhertz (or given by the user as `spacing`) and evaluates the
    autocorrelation at each step.

    The correlation (numpy.correlate) is typically given as:

        C[x, y] = sum( x * conj(y) ) .

    The autocorrelation power of a full spectrum with itself is then

        C = sum(s * s),

    where s is a window of the signal-to-noise spectrum.
    Because of the method of this calculation, we need to first
    rescale the power by subtracting its mean, placing its mean around 0. This
    decreases the noise levels in the ACF, as the autocorrelation of the noise
    with itself will be close to zero.

    In order to evaluate where the correlation power is highest (indicative
    of the power excess of the modes) we calculate the Mean Collapsed
    Correlation (MCC, see Kiefer 2013, Viani et al. 2019) as

        MCC = (sum(|C|) - 1) / nlags ,

    where C is the autocorrelation power at a given central freqeuncy, and
    nlags is the number of lags in the autocorrelation.

    The MCC metric is covolved with an Astropy Gaussian 1D Kernel with a
    standard deviation of 1/5th of the window size to smooth it. The
    frequency that results in the highest value of the smoothed MCC is the
    detected numax.

    NOTE: This method is not robust against large peaks in the spectrum (due
    to e.g. spacecraft rotation), nor is it robust in the case of low signal
    to noise (such as for single sector TESS data). Exercise caution when
    using this module!

    NOTE: This function is intended for use with solar like Main Sequence
    and Red Giant Branch oscillators only.

    Parameters
    ----------
    numaxs : array-like
        An array of numaxs at which to evaluate the autocorrelation. If
        none is given, a sensible range will be chosen. If no units are
        given it is assumed to be in the same units as the periodogram
        frequency.
    window_width : int or float
        The width of the autocorrelation window around each central
        frequency in 'numaxs'. If none is given, a sensible value will be
        chosen. If no units are given it is assumed to be in the same units
        as the periodogram frequency.
    spacing : int or float
        The spacing between central frequencies (numaxs) at which the
        autocorrelation is evaluated. If none is given, a sensible value
        will be assumed. If no units are given it is assumed to be in the
        same units as the periodogram frequency.

    Returns
    -------
    numax : `.SeismologyQuantity`
        The numax of the periodogram. In the units of the periodogram object
        frequency.
    """
    # Detect whether the frequency grid is evenly-spaced
    if not periodogram._is_evenly_spaced():
        raise ValueError(
            "the ACF 2D method requires that the periodogram "
            "has a grid of uniformly spaced frequencies."
        )

    # Calculate the window_width size

    # C: What is this doing? Why have these values been picked? This function is slow.
    if window_width is None:
        if u.Quantity(periodogram.frequency[-1], u.microhertz) > u.Quantity(
            500.0, u.microhertz
        ):
            window_width = (
                u.Quantity(250.0, u.microhertz).to(periodogram.frequency.unit).value
            )
        else:
            window_width = (
                u.Quantity(25.0, u.microhertz).to(periodogram.frequency.unit).value
            )

    # Calculate the spacing size
    if spacing is None:
        if u.Quantity(periodogram.frequency[-1], u.microhertz) > u.Quantity(
            500.0, u.microhertz
        ):
            spacing = (
                u.Quantity(10.0, u.microhertz).to(periodogram.frequency.unit).value
            )
        else:
            spacing = u.Quantity(1.0, u.microhertz).to(periodogram.frequency.unit).value

    # Run some checks on the inputs
    window_width = u.Quantity(window_width, periodogram.frequency.unit).value
    spacing = u.Quantity(spacing, periodogram.frequency.unit).value
    if numaxs is None:
        numaxs = np.arange(
            np.ceil(np.nanmin(periodogram.frequency.value)) + window_width / 2,
            np.floor(np.nanmax(periodogram.frequency.value)) - window_width / 2,
            spacing,
        )
    numaxs = u.Quantity(numaxs, periodogram.frequency.unit).value
    if not hasattr(numaxs, "__iter__"):
        numaxs = np.asarray([numaxs])

    fs = np.median(np.diff(periodogram.frequency.value))
    # Perform checks on spacing and window_width
    for var, label in zip(
        [np.asarray(window_width), np.asarray(spacing)], ["window_width", "spacing"]
    ):
        if (var < fs).any():
            raise ValueError(
                "You can't have {} smaller than the "
                "frequency separation!".format(label)
            )
        if (
            var > (periodogram.frequency[-1].value - periodogram.frequency[0].value)
        ).any():
            raise ValueError(
                "You can't have {} wider than the entire "
                "power spectrum!".format(label)
            )
        if (var < 0).any():
            raise ValueError("Please pass an entirely positive {}.".format(label))

    # Perform checks on numaxs
    if any(numaxs < fs):
        raise ValueError(
            "A custom range of numaxs can not extend below " "a single frequency bin."
        )
    if any(numaxs > np.nanmax(periodogram.frequency.value)):
        raise ValueError(
            "A custom range of numaxs can not extend above "
            "the highest frequency value in the periodogram."
        )

    # We want to find the numax which returns in the highest autocorrelation
    # power, rescaled based on filter width
    fs = np.median(np.diff(periodogram.frequency.value))

    metric = np.zeros(len(numaxs))
    acf2d = np.zeros([int(window_width / 2 / fs) * 2, len(numaxs)])
    for idx, numax in enumerate(numaxs):
        acf = utils.autocorrelate(
            periodogram, numax, window_width=window_width, frequency_spacing=fs
        )  # Return the acf at this numax
        acf2d[:, idx] = acf  # Store the 2D acf
        metric[idx] = (np.sum(np.abs(acf)) - 1) / len(
            acf
        )  # Store the max acf power normalised by the length

    # Smooth the data to find the peak
    # Gaussian1D kernel takes a standard deviation in unitless indices. A stddev
    # of sqrt(len(numaxs) will result in a smoothing kernel that works for all
    # resolutions of numax.
    if len(numaxs) > 10:
        g = Gaussian1DKernel(stddev=np.sqrt(len(numaxs)))
        metric_smooth = convolve(metric, g, boundary="extend")
    else:
        metric_smooth = metric

    # The highest value of the metric corresponds to numax
    best_numax = numaxs[np.argmax(metric_smooth)]
    best_numax = u.Quantity(best_numax, periodogram.frequency.unit)

    # Create and return the object containing the result and diagnostics
    diagnostics = {
        "numaxs": numaxs,
        "acf2d": acf2d,
        "window_width": window_width,
        "metric": metric,
        "metric_smooth": metric_smooth,
    }
    result = SeismologyQuantity(
        best_numax,
        name="numax",
        method="ACF2D",
        diagnostics=diagnostics,
        diagnostics_plot_method=diagnose_numax_acf2d,
    )
    return result


def diagnose_numax_acf2d(numax, periodogram):
    """Returns a diagnostic plot which elucidates how numax was estimated.

    [1] The SNRPeriodogram plotted with a red line indicating the estimated
    numax value.

    [2] An image showing the 2D autocorrelation. On the y-axis is the
    frequency lag of the autocorrelation window. The width of the window is
    equal to `window_width`, and the spacing between lags is equal to
    `numax_spacing`. On the x-axis is the central frequency at which the
    autocorrelation was calculated. In the z-axis is the unitless
    autocorrelation power. Shown in red is the estimated numax.

    [3] The Mean Collapsed Correlation (MCC, see Viani et al. 2019) against
    central frequency at which the MCC was calculated. Shown in red is the
    estimated numax. Shown in blue is the MCC convolved with a Gaussian
    smoothing kernel with a standard deviation of 1/5th the window size.

    For details on the numax estimation, see the `estimate_numax()` function.
    The calculation performed is identical

    Parameters
    ----------
    numax : `.SeismologyResult` object
        The object returned by `estimate_numax_acf2d()`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        The matplotlib axes object.
    """
    with plt.style.context(MPLSTYLE):
        fig, ax = plt.subplots(3, sharex=True, figsize=(8.485, 12))
        periodogram.plot(ax=ax[0], label="")
        ax[0].axvline(
            numax.value,
            c="r",
            linewidth=2,
            alpha=0.4,
            label="{} = {:7.5} {}".format(
                r"$\nu_{\rm max}$",
                numax.value,
                periodogram.frequency.unit.to_string("latex"),
            ),
        )
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel("")
        ax[0].text(
            0.05,
            0.9,
            "Input Power Spectrum",
            horizontalalignment="left",
            transform=ax[0].transAxes,
            fontsize=15,
        )

        vmin = np.nanpercentile(numax.diagnostics["acf2d"], 5)
        vmax = np.nanpercentile(numax.diagnostics["acf2d"], 95)
        ax[1].pcolormesh(
            numax.diagnostics["numaxs"],
            np.linspace(
                0,
                numax.diagnostics["window_width"],
                num=numax.diagnostics["acf2d"].shape[0],
            ),
            numax.diagnostics["acf2d"],
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
        )
        ax[1].set_ylabel(
            r"Frequency lag [{}]".format(periodogram.frequency.unit.to_string("latex"))
        )
        ax[1].axvline(numax.value, c="r", linewidth=2, alpha=0.4)
        ax[1].text(
            0.05,
            0.9,
            "2D AutoCorrelation",
            horizontalalignment="left",
            transform=ax[1].transAxes,
            fontsize=13,
        )

        ax[2].plot(numax.diagnostics["numaxs"], numax.diagnostics["metric"])
        ax[2].plot(
            numax.diagnostics["numaxs"],
            numax.diagnostics["metric_smooth"],
            lw=2,
            alpha=0.7,
            label="Smoothed Metric",
        )
        ax[2].set_xlabel(
            "Frequency [{}]".format(periodogram.frequency.unit.to_string("latex"))
        )
        ax[2].set_ylabel(r"Correlation Metric")

        ax[2].axvline(numax.value, c="r", linewidth=2, alpha=0.4)
        ax[2].text(
            0.05,
            0.9,
            "Correlation Metric",
            horizontalalignment="left",
            transform=ax[2].transAxes,
            fontsize=13,
        )
        ax[2].legend(loc="upper right")
        ax[2].set_xlim(numax.diagnostics["numaxs"][0], numax.diagnostics["numaxs"][-1])
        plt.subplots_adjust(hspace=0, wspace=0)
    return ax
