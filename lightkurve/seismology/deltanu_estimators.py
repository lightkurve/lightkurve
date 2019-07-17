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
    diagnostics = {'lags':lags, 'acf':acf, 'peaks':peaks, 'sel':sel}
    result = SeismologyQuantity(best_deltanu,
                                name="deltanu",
                                method="Mosser & Appourchaux 2009",
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
        fig, ax = plt.subplots()
        # ax.plot(lags, acf/acf[0])
        ax.plot(deltanu.diagnostics['lags'][1:], deltanu.diagnostics['acf'][1:])
        ax.set_xlabel("Frequency Lag [{}]".format(periodogram.frequency.unit.to_string('latex')))
        ax.set_ylabel(r'Scaled Correlation')
        ax.axvline(deltanu.value,c='r', linewidth=2,alpha=.4)
        ax.set_title(r'Scaled Correlation vs Lag for a given $\nu_{\rm max}$')

        axin = inset_axes(ax, width="50%",height="50%", loc="upper right")
        axin.set_yticks([])
        axin.plot(deltanu.diagnostics['lags'][deltanu.diagnostics['sel']],
                  deltanu.diagnostics['acf'][deltanu.diagnostics['sel']])
        axin.scatter(deltanu.diagnostics['lags'][deltanu.diagnostics['sel']][deltanu.diagnostics['peaks']],
                     deltanu.diagnostics['acf'][deltanu.diagnostics['sel']][deltanu.diagnostics['peaks']],
                     c='r', s=5)
        axin.axvline(deltanu.value,c='r', linewidth=2,alpha=.4,
            label=r'{:.1f} {}'.format(deltanu.value,
                                      periodogram.frequency.unit.to_string('latex')))
        axin.legend(loc='best')
        return ax
