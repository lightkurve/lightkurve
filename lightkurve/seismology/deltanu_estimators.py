"""Helper functions for estimating deltanu from periodograms.

Functions in this module should be named "estimate_deltanu_methodname()".
"""
from __future__ import division, print_function

import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy import units as u

from .utils import SeismologyQuantity
from . import utils
from .. import MPLSTYLE


def estimate_deltanu_acf(periodogram, numax):
    """
    Helper function to perform the deltanu estimation for both the
    `estimate_deltanu()` and `plot_deltanu_diagnostics()` functions.

    For details, see the `estimate_deltanu()` function.
    """

    # Run some checks on the passed in numaxs
    # Ensure input numax is in the correct units
    numax = u.Quantity(numax, periodogram.frequency.unit)
    fs = np.median(np.diff(periodogram.frequency.value))
    if numax.value < fs:
        raise ValueError("The input numax can not be lower than"
                        " a single frequency bin.")
    if numax.value > np.nanmax(periodogram.frequency.value):
        raise ValueError("The input numax can not be higher than"
                        "the highest frequency value in the periodogram.")

    # Calculate deltanu using the method by Stello et al. 2009
    # Make sure that this relation only ever happens in microhertz space
    deltanu_emp = u.Quantity((0.294 * u.Quantity(numax, u.microhertz).value ** 0.772)*u.microhertz,
                        periodogram.frequency.unit).value

    window = 2*int(np.floor(utils.get_fwhm(periodogram, numax.value)))
    aacf = utils.autocorrelate(periodogram, numax=numax.value, window=window)
    acf = (np.abs(aacf**2)/np.abs(aacf[0]**2)) / (3/(2*len(aacf)))
    fs = np.median(np.diff(periodogram.frequency.value))
    lags = np.linspace(0., len(acf)*fs, len(acf))

    #Select a 25% region region around the empirical deltanu
    sel = (lags > deltanu_emp - .25*deltanu_emp) & (lags < deltanu_emp + .25*deltanu_emp)

    #Run a peak finder on this region
    peaks, _ = find_peaks(acf[sel], distance=np.floor(deltanu_emp/2. / fs))

    #Select the peak closest to the empirical value
    best_deltanu_value = lags[sel][peaks][np.argmin(np.abs(lags[sel][peaks] - deltanu_emp))]
    best_deltanu = u.Quantity(best_deltanu_value, periodogram.frequency.unit)
    diagnostics = {'lags':lags, 'acf':acf, 'peaks':peaks, 'sel':sel, 'numax':numax, 'deltanu_emp':deltanu_emp}
    result = SeismologyQuantity(best_deltanu,
                                name="deltanu",
                                method="ACF",
                                diagnostics=diagnostics,
                                diagnostics_plot_method=diagnose_deltanu_acf)
    return result


def diagnose_deltanu_acf(deltanu, periodogram):
    """Returns a diagnostic plot which elucidates how deltanu was estimated.

    [1] Scaled correlation metric vs frequecy lag of the autocorrelation
    window, with inset close up on the determined deltanu and a line
    indicating the determined deltanu.

    For details on the deltanu estimation, see the `estimate_deltanu()`
    function. The calculation performed is identical.

    NOTE: When plotting , we exclude the first frequency lag bin, to
    make the relevant features on the plot clearer.

    Parameters
    -----------
    deltanu : `SeismologyResult` object
        The object returned by `estimate_deltanu_acf()`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    """
    with plt.style.context(MPLSTYLE):
        fig, axs = plt.subplots(2, figsize=(8.485, 8))

        ax = axs[0]
        periodogram.plot(ax=ax, label='')
        ax.axvline(deltanu.diagnostics['numax'].value, c='r', linewidth=1, alpha=.4, ls=':')
        ax.text(deltanu.diagnostics['numax'].value, periodogram.power.value.max()*0.75, 'numax'+' ({0:7.5})'.format(deltanu.diagnostics['numax']), rotation=90, ha='right', color='r', alpha=0.5, fontsize=8)
        ax.text(.025, .9, 'Input Power Spectrum',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontsize=11)


        window = 2*int(np.floor(utils.get_fwhm(periodogram, deltanu.diagnostics['numax'].value)))
        frequency_spacing = np.median(np.diff(periodogram.frequency.value))
        spread = int(window/2/frequency_spacing)                           # Find the spread in indices


        a = periodogram.frequency.value[np.argmin(np.abs(periodogram.frequency.value - deltanu.diagnostics['numax'].value)) + spread]
        b = periodogram.frequency.value[np.argmin(np.abs(periodogram.frequency.value - deltanu.diagnostics['numax'].value)) - spread]
        ax.axvline(a, c='r', linewidth=2, alpha=.4, ls='--')
        ax.axvline(b, c='r', linewidth=2, alpha=.4, ls='--')
        h = periodogram.power.value.max() * 0.9

        ax.annotate("", xy=(a, h), xytext=(a - (a-b), h),
                        arrowprops=dict(arrowstyle="<->", color='r', alpha=0.5), va='bottom')
        ax.text(a - (a-b)/2, h, "FWHM", color='r', alpha=0.7, fontsize=10, va='bottom', ha='center')


        ax = axs[1]
        # ax.plot(lags, acf/acf[0])
        ax.plot(deltanu.diagnostics['lags'][1:], deltanu.diagnostics['acf'][1:])
        ax.set_xlabel("Frequency Lag [{}]".format(periodogram.frequency.unit.to_string('latex')))
        ax.set_ylabel(r'Scaled Auto Correlation', fontsize=11)

#        ax.set_title(r'Scaled Correlation vs Lag for a given $\nu_{\rm max}$')

        axin = inset_axes(ax, width="50%",height="50%", loc="upper right")
        axin.set_yticks([])
        axin.plot(deltanu.diagnostics['lags'][deltanu.diagnostics['sel']],
                  deltanu.diagnostics['acf'][deltanu.diagnostics['sel']])
        axin.scatter(deltanu.diagnostics['lags'][deltanu.diagnostics['sel']][deltanu.diagnostics['peaks']],
                     deltanu.diagnostics['acf'][deltanu.diagnostics['sel']][deltanu.diagnostics['peaks']],
                     c='r', s=5)
        ax.axvline(deltanu.value,c='r', linewidth=2,alpha=.4,
            label=r'Measured deltanu {:.1f} {}'.format(deltanu.value,
                                      periodogram.frequency.unit.to_string('latex')))
        ax.axvline(deltanu.diagnostics['deltanu_emp'],c='b', linewidth=2,alpha=.4, ls='--',
                    label=r'Empiracal deltanu {:.1f} {}'.format(deltanu.diagnostics['deltanu_emp'],
                                  periodogram.frequency.unit.to_string('latex')))
        axin.axvline(deltanu.value,c='r', linewidth=2,alpha=.4)
        axin.axvline(deltanu.diagnostics['deltanu_emp'],c='b', linewidth=2,alpha=.4, ls='--')
        ax.text(.025, .9, 'Scaled Auto Correlation Within FWHM',
                    horizontalalignment='left',
                    transform=ax.transAxes, fontsize=11)

        ax.legend(loc='lower right', fontsize=10)
        return ax
