"""Defines the Seismology class."""
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.units import cds

from .. import MPLSTYLE
from . import utils, stellar_estimators
from ..periodogram import SNRPeriodogram
from ..utils import LightkurveWarning, validate_method

from scipy.signal import find_peaks

log = logging.getLogger(__name__)

__all__ = ['Seismology']


class Seismology(object):
    """Enables astroseismic quantities to be estimated from periodograms.

    This class provides easy access to methods to estimate numax, deltanu, radius,
    mass, and logg, and stores them on its tray for easy diagnostic plotting.

    Examples
    --------
    Download the TESS light curve for HIP 116158:

        >>> import lightkurve as lk
        >>> lc = lk.search_lightcurvefile("HIP 116158", sector=2).download().PDCSAP_FLUX
        >>> lc = lc.normalize().remove_nans().remove_outliers()

    Create a Lomb-Scargle periodogram:

        >>> pg = lc.to_periodogram(normalization='psd', minimum_frequency=100, maximum_frequency=800)

    Create a Seismology object and use it to estimate parameters:

        >>> seismology = pg.flatten().to_seismology()
        >>> seismology.estimate_numax()
        numax: 415.00 uHz (method: ACF2D)
        >>> seismology.estimate_deltanu()
        deltanu: 28.78 uHz (method: ACF2D)
        >>> seismology.estimate_radius(teff=5080)
        radius: 2.78 solRad (method: Uncorrected Scaling Relations)

    Parameters
    ----------
    periodogram : `~lightkurve.periodogram.Periodogram` object
        Periodogram to be analyzed. Must be background-corrected,
        e.g. using `periodogram.flatten()`.
    """
    def __init__(self, periodogram):
        if not isinstance(periodogram, SNRPeriodogram):
            warnings.warn("Seismology received a periodogram which does not appear "
                          "to have been background-corrected. Please consider calling "
                          "`periodogram.flatten()` prior to extracting seismological parameters.",
                          LightkurveWarning)
        self.periodogram = periodogram

    def __repr__(self):
        attrs = np.asarray(['numax', 'deltanu', 'mass', 'radius', 'logg'])
        tray = np.asarray([hasattr(self, attr) for attr in attrs])
        if tray.sum() == 0:
            tray_str = " - no values have been computed so far."
        else:
            tray_str = " - computed values:\n * " + "\n * ".join([getattr(self, attr).__repr__() for attr in attrs[tray]])
        return 'Seismology(ID: {}){}'.format(self.periodogram.label, tray_str)

    @staticmethod
    def from_lightcurve(lc, **kwargs):
        """Returns a `Seismology` object given a `~lightkurve.lightcurve.LightCurve` object."""
        log.info("Building a Seismology object directly from a light curve "
                 "uses default periodogram parameters. For further tuneability, "
                 "create a periodogram object first, using `to_periodogram`.")
        return Seismology(periodogram=lc.normalize().remove_nans().fill_gaps().to_periodogram(**kwargs).flatten())

    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError("You need to call `Seismology.estimate_deltanu()` first.")
        return deltanu


    def plot_echelle(self, deltanu=None, numax=None,
                     minimum_frequency=None, maximum_frequency=None,
                     smooth_filter_width=.1, ax=None,
                     scale='linear', cmap='Blues'):
        """Plots an echelle diagram of the periodogram by stacking the
        periodogram in slices of deltanu.

        Modes of equal radial degree should appear approximately vertically aligned.
        If no structure is present, you are likely dealing with a faulty deltanu
        value or a low signal to noise case.

        This method is adapted from work by Daniel Hey & Guy Davies.

        Parameters
        ----------
        deltanu : float
            Value for the large frequency separation of the seismic mode
            frequencies in the periodogram. Assumed to have the same units as
            the frequencies, unless given an Astropy unit.
            Is assumed to be in the same units as frequency if not given a unit.
        numax : float
            Value for the frequency of maximum oscillation. If a numax is
            passed, a suitable range one FWHM of the mode envelope either side
            of the will be shown. This is overwritten by custom frequency ranges.
            Is assumed to be in the same units as frequency if not given a unit.
        minimum_frequency : float
            The minimum frequency at which to display the echelle
            Is assumed to be in the same units as frequency if not given a unit.
        maximum_frequency : float
            The maximum frequency at which to display the echelle.
            Is assumed to be in the same units as frequency if not given a unit.
        smooth_filter_width : float
            If given a value, will smooth periodogram used to plot the echelle
            diagram using the periodogram.smooth(method='boxkernel') method with
            a filter width of `smooth_filter_width`. This helps visualise the
            echelle diagram. Is assumed to be in the same units as the
            periodogram frequency.
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        scale: str
            Set z axis to be "linear" or "log". Default is linear.
        cmap : str
            The name of the matplotlib colourmap to use in the echelle diagram.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        numax = self._validate_numax(numax)
        if not hasattr(numax, 'unit'):
            numax = numax * self.periodogram.frequency.unit
        deltanu = self._validate_deltanu(deltanu)
        if not hasattr(deltanu, 'unit'):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
            power = pgsmooth.power     # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable
            power = self.periodogram.power     # Makes code below more readable

        fmin = freq[0]
        fmax = freq[-1]

        # Check for any superfluous input
        if (numax is not None) & (any([a is not None for a in [minimum_frequency, maximum_frequency]])):
            warnings.warn("You have passed both a numax and a frequency limit. "
                          "The frequency limit will override the numax input.",
                          LightkurveWarning)

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, freq.unit).value
            if numax > freq[-1].value:
                raise ValueError("You can't pass in a numax outside the"
                                "frequency range of the periodogram.")

            fwhm = utils.get_fwhm(self.periodogram, numax)

            fmin = numax - 2*fwhm
            if fmin < freq[0].value:
                fmin = freq[0].value

            fmax = numax + 2*fwhm
            if fmax > freq[-1].value:
                fmax = freq[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin =  u.Quantity(minimum_frequency, freq.unit).value
            if fmin > freq[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                 "frequency range of the periodogram.")

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, freq.unit).value
            if fmax > freq[-1].value:
                raise ValueError("You can't pass in a limit outside the"
                                 "frequency range of the periodogram.")

        # Make sure fmin and fmax are Quantities or code below will break
        fmin = u.Quantity(fmin, freq.unit)
        fmax = u.Quantity(fmax, freq.unit)

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < freq[-1] - 1.5*deltanu:
            fmax += deltanu

        fs = np.median(np.diff(freq))
        x0 = int(freq[0] / fs)

        ff = freq[int(fmin/fs)-x0:int(fmax/fs)-x0] # Selected frequency range
        pp = power[int(fmin/fs)-x0:int(fmax/fs)-x0] # Power range

        # Reshape the power into n_rows of n_columns
        #  When modulus ~ zero, deltanu divides into frequency without remainder
        mod_zeros = find_peaks( -1.0*(ff % deltanu) )[0]

        # The bottom left corner of the plot is the lowest frequency that
        # divides into deltanu with almost zero remainder
        start = mod_zeros[0]

        # The top left corner of the plot is the highest frequency that
        # divides into deltanu with almost zero remainder.  This index is the
        # approximate end, because we fix an integer number of rows and columns
        approx_end = mod_zeros[-1]

        # The number of rows is the number of times you can partition your
        #  frequency range into chunks of size deltanu, start and ending at
        #  frequencies that divide nearly evenly into deltanu
        n_rows = len(mod_zeros) - 1

        # The number of columns is the total number of frequency points divided
        #  by the number of rows, floor divided to the nearest integer value
        n_columns =  int( (approx_end - start) / n_rows )

        # The exact end point is therefore the ncolumns*nrows away from the start
        end = start + n_columns*n_rows

        ep = np.reshape(pp[start : end], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        #Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[start : end], (n_rows, n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        y_f = (ef[:,0])

        #Plot the echelle diagram
        with plt.style.context(MPLSTYLE):
            if ax is None:
                fig, ax = plt.subplots()
            extent = (x_f[0].value, x_f[-1].value, y_f[0].value, y_f[-1].value)
            figsize = plt.rcParams['figure.figsize']
            a = figsize[1] / figsize[0]
            b = (extent[3] - extent[2]) / (extent[1] - extent[0])
            vmin = np.nanpercentile(ep.value, 1)
            vmax = np.nanpercentile(ep.value, 99)

            im = ax.imshow(ep.value, cmap=cmap, aspect=a/b, origin='lower',
                        extent=extent, vmin=vmin, vmax=vmax)

            cbar = plt.colorbar(im, ax=ax, extend='both', pad=.01)


            if isinstance(self.periodogram, SNRPeriodogram):
                ylabel = 'Signal to Noise Ratio (SNR)'
            elif self.periodogram.power.unit == cds.ppm:
                ylabel = "Amplitude [{}]".format(self.periodogram.power.unit.to_string('latex'))
            else:
                ylabel = "Power Spectral Density [{}]".format(self.periodogram.power.unit.to_string('latex'))

            if scale == 'log':
                ylabel = 'log10('+ylabel+')'

            cbar.set_label(ylabel)
            ax.set_xlabel(r'Frequency mod. {:.2f}'.format(deltanu))
            ax.set_ylabel(r'Frequency [{}]'.format(freq.unit.to_string('latex')))
            ax.set_title('Echelle diagram for {}'.format(self.periodogram.label))

        return ax


    def estimate_numax(self, method="acf2d", **kwargs):
        """Returns the frequency of the peak of the seismic oscillation modes envelope.

        At present, the only method supported is based on using a
        2D autocorrelation function (ACF2D).  This method is implemented by the
        `~lightkurve.seismology.estimate_numax_acf2d` function which accepts
        the parameters `numaxs`, `window_width`, and `spacing`.
        For details and literature references, please read the detailed
        docstring of this function by typing ``lightkurve.seismology.estimate_numax_acf2d?``
        in a Python terminal or notebook.

        Parameters
        ----------
        method : str
            Method to use. Only ``"acf2d"`` is supported at this time.

        Returns
        -------
        numax : `~lightkurve.seismology.SeismologyQuantity`
            Numax of the periodogram, including details on the units and method.
        """
        method = validate_method(method, supported_methods=["acf2d"])
        if method == "acf2d":
            from .numax_estimators import estimate_numax_acf2d
            result = estimate_numax_acf2d(self.periodogram, **kwargs)
        self.numax = result
        return result

    def diagnose_numax(self, numax=None):
        """Create diagnostic plots showing how numax was estimated."""
        numax = self._validate_numax(numax)
        return numax.diagnostics_plot_method(numax, self.periodogram)

    def estimate_deltanu(self, method='acf2d', numax=None):
        """Returns the average value of the large frequency spacing, DeltaNu,
        of the seismic oscillations of the target.

        At present, the only method supported is based on using an
        autocorrelation function (ACF2D).  This method is implemented by the
        `~lightkurve.seismology.estimate_deltanu_acf2d` function which requires
        the parameter `numax`. For details and literature references, please
        read the detailed docstring of this function by typing
        ``lightkurve.seismology.estimate_deltanu_acf2d?`` in a Python terminal or notebook.

        Parameters
        ----------
        method : str
            Method to use. Only ``"acf2d"`` is supported at this time.

        Returns
        -------
        deltanu : `~lightkurve.seismology.SeismologyQuantity`
            DeltaNu of the periodogram, including details on the units and method.
        """
        method = validate_method(method, supported_methods=["acf2d"])
        numax = self._validate_numax(numax)

        if method == "acf2d":
            from .deltanu_estimators import estimate_deltanu_acf2d
            result = estimate_deltanu_acf2d(self.periodogram, numax=numax)

        self.deltanu = result
        return result

    def diagnose_deltanu(self, deltanu=None):
        """Create diagnostic plots showing how numax was estimated."""
        deltanu = self._validate_deltanu(deltanu)
        return deltanu.diagnostics_plot_method(deltanu, self.periodogram)

    def estimate_radius(self, teff, numax=None, deltanu=None):
        """Returns a stellar radius estimate based on the scaling relations.

        This method is implemented by the `~lightkurve.seismology.estimate_radius` function.
        For details and literature references, please read the detailed
        docstring of this function by typing ``lightkurve.seismology.estimate_radius?``.

        Returns
        -------
        radius : `~lightkurve.seismology.SeismologyQuantity`
            Stellar radius estimate.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        result = stellar_estimators.estimate_radius(numax, deltanu, teff)
        self.radius = result
        return result

    def estimate_mass(self, teff, numax=None, deltanu=None):
        """Returns a stellar mass estimate based on the scaling relations.

        This method is implemented by the `~lightkurve.seismology.estimate_mass` function.
        For details and literature references, please read the detailed
        docstring of this function by typing ``lightkurve.seismology.estimate_mass?``.

        Returns
        -------
        mass : `~lightkurve.seismology.SeismologyQuantity`
            Stellar mass estimate.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        result = stellar_estimators.estimate_mass(numax, deltanu, teff)
        self.mass = result
        return result

    def estimate_logg(self, teff, numax=None):
        """Returns a surface gravity estimate based on the scaling relations.

        This method is implemented by the `~lightkurve.seismology.estimate_logg` function.
        For details and literature references, please read the detailed
        docstring of this function by typing ``lightkurve.seismology.estimate_logg?``.

        Returns
        -------
        logg : `~lightkurve.seismology.SeismologyQuantity`
            Stellar surface gravity estimate.
        """
        numax = self._validate_numax(numax)
        result = stellar_estimators.estimate_logg(numax, teff)
        self.logg = result
        return result
