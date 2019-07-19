"""Helper functions for estimating numax from periodograms.

Functions in this module should be named "estimate_numax_methodname()".
"""
import numpy as np
from matplotlib import pyplot as plt

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
from astropy import units as u

from .. import MPLSTYLE
from . import utils
from .utils import SeismologyQuantity


def estimate_numax_acf(periodogram, numaxs=None, window_width=None, spacing=None):
    """Estimate the frequency at maximum power from a periodogram using
    the autocorrelation method.
    """
    # Calculate the window_width size

    #C: What is this doing? Why have these values been picked? This function is slow.
    if window_width is None:
        if u.Quantity(periodogram.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
            window_width = u.Quantity(250., u.microhertz).to(periodogram.frequency.unit).value
        else:
            window_width = u.Quantity(25., u.microhertz).to(periodogram.frequency.unit).value

    # Calculate the spacing size
    if spacing is None:
        if u.Quantity(periodogram.frequency[-1], u.microhertz) > u.Quantity(500., u.microhertz):
            spacing = u.Quantity(10., u.microhertz).to(periodogram.frequency.unit).value
        else:
            spacing = u.Quantity(1., u.microhertz).to(periodogram.frequency.unit).value

    # Run some checks on the inputs
    window_width = u.Quantity(window_width, periodogram.frequency.unit).value
    spacing = u.Quantity(spacing, periodogram.frequency.unit).value
    if numaxs is None:
        numaxs = np.arange(np.ceil(np.nanmin(periodogram.frequency.value)) + window_width/2,
                    np.floor(np.nanmax(periodogram.frequency.value)) - window_width/2,
                    spacing)
    numaxs = u.Quantity(numaxs, periodogram.frequency.unit).value
    if not hasattr(numaxs, '__iter__'):
        numaxs = np.asarray([numaxs])

    fs = np.median(np.diff(periodogram.frequency.value))
    # Perform checks on spacing and window_width
    for var, label in zip([np.asarray(window_width), np.asarray(spacing)],
                          ['window_width', 'spacing']):
        if (var < fs).any():
            raise ValueError("You can't have {} smaller than the "
                            "frequency separation!".format(label))
        if (var > (periodogram.frequency[-1].value - periodogram.frequency[0].value)).any():
            raise ValueError("You can't have {} wider than the entire "
                            "power spectrum!".format(label))
        if (var < 0).any():
            raise ValueError("Please pass an entirely positive {}.".format(label))

    # Perform checks on numaxs
    if any(numaxs < fs):
        raise ValueError("A custom range of numaxs can not extend below "
                        "a single frequency bin.")
    if any(numaxs > np.nanmax(periodogram.frequency.value)):
        raise ValueError("A custom range of numaxs can not extend above "
                        "the highest frequency value in the periodogram.")

    # We want to find the numax which returns in the highest autocorrelation
    # power, rescaled based on filter width
    fs = np.median(np.diff(periodogram.frequency.value))

    metric = np.zeros(len(numaxs))
    acf2d = np.zeros([int(window_width/2/fs)*2,len(numaxs)])
    for idx, numax in enumerate(numaxs):
        acf = utils.autocorrelate(periodogram, numax, window_width=window_width, frequency_spacing=fs)      #Return the acf at this numax
        acf2d[:, idx] = acf                                     #Store the 2D acf
        metric[idx] = (np.sum(np.abs(acf)) - 1 ) / len(acf)  #Store the max acf power normalised by the length

    # Smooth the data to find the peak
    # Gaussian1D kernel takes a standard deviation in unitless indices. A stddev
    # of sqrt(len(numaxs) will result in a smoothing kernel that works for all
    # resolutions of numax.
    if len(numaxs) > 10:
        g = Gaussian1DKernel(stddev=np.sqrt(len(numaxs)))
        metric_smooth = convolve(metric, g, boundary='extend')
    else:
        metric_smooth = metric

    # The highest value of the metric corresponds to numax
    best_numax = numaxs[np.argmax(metric_smooth)]
    best_numax = u.Quantity(best_numax, periodogram.frequency.unit)

    # Create and return the object containing the result and diagnostics
    diagnostics = {'numaxs':numaxs, 'acf2d':acf2d, 'window_width':window_width,
                   'metric':metric, 'metric_smooth': metric_smooth}
    result = SeismologyQuantity(best_numax,
                                name="numax",
                                method="ACF",
                                diagnostics=diagnostics,
                                diagnostics_plot_method=diagnose_numax_acf)
    return result


def diagnose_numax_acf(numax, periodogram):
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

    Parameters:
    -----------
    numax : `SeismologyResult` object
        The object returned by `estimate_numax_acf()`.

    Returns:
    --------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    with plt.style.context(MPLSTYLE):
        fig, ax = plt.subplots(3, sharex=True, figsize=(8.485, 12))
        periodogram.plot(ax=ax[0], label='')
        ax[0].axvline(numax.value, c='r', linewidth=2, alpha=.4)
        ax[0].text(numax.value, periodogram.power.value.max()*0.75, '{:7.5}'.format(numax.value), rotation=90, ha='right', color='r', alpha=0.5)
        ax[0].set_xlabel('')
        ax[0].text(.05, .9, 'Input Power Spectrum',
                    horizontalalignment='left',
                    transform=ax[0].transAxes, fontsize=15)

        windowarray = np.linspace(0, numax.diagnostics['window_width'],
                                  num=numax.diagnostics['acf2d'].shape[1])
        extent = (numax.diagnostics['numaxs'][0],
                  numax.diagnostics['numaxs'][-1],
                  windowarray[0],
                  windowarray[-1])
        figsize = [8.485, 4]
        a = figsize[1] / figsize[0]
        b = (extent[3] - extent[2]) / (extent[1] - extent[0])
        vmin = np.nanpercentile(numax.diagnostics['acf2d'], 5)
        vmax = np.nanpercentile(numax.diagnostics['acf2d'], 95)
#        print(windowarray.shape)
#        print(numax.diagnostics['numaxs'].shape)
#        print(numax.diagnostics['acf2d'].shape)
#        print(periodogram.frequency.value.shape)
        im = ax[1].pcolormesh(numax.diagnostics['numaxs'], np.linspace(0, numax.diagnostics['window_width'], num=numax.diagnostics['acf2d'].shape[0]),
                                    numax.diagnostics['acf2d'], cmap='Blues', vmin=vmin, vmax=vmax)
#        plt.colorbar(im, ax=ax[1], orientation='horizontal')
        ax[1].set_ylabel(r'Frequency lag [{}]'.format(periodogram.frequency.unit.to_string('latex')))
        ax[1].axvline(numax.value, c='r', linewidth=2, alpha=.4)
        ax[1].text(.05, .9, '2D AutoCorrelation',
                    horizontalalignment='left',
                    transform=ax[1].transAxes, fontsize=13)

        ax[2].plot(numax.diagnostics['numaxs'], numax.diagnostics['metric'])
        ax[2].plot(numax.diagnostics['numaxs'], numax.diagnostics['metric_smooth'], lw=2, alpha=0.7)
        ax[2].set_xlabel("Frequency [{}]".format(periodogram.frequency.unit.to_string('latex')))
        ax[2].set_ylabel(r'Correlation Metric')


        ax[2].axvline(numax.value, c='r', linewidth=2, alpha=.4)
        ax[2].text(.05, .9, 'Correlation Metric',
                            horizontalalignment='left',
                            transform=ax[2].transAxes, fontsize=13)
        ax[2].set_xlim(numax.diagnostics['numaxs'][0], numax.diagnostics['numaxs'][-1])
        plt.subplots_adjust(hspace=0, wspace=0)
    return ax
