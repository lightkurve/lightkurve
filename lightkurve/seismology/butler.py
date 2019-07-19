"""Defines the SeismologyButler class.

TODO
----
* Consider putting Teff in repr of stellar parameters.
* Errors are not yet passed to stellar_estimator functions.
* Clean up docstrings and plots.
"""
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.units import cds

from .. import MPLSTYLE
from . import utils, stellar_estimators
from ..periodogram import LombScarglePeriodogram, SNRPeriodogram
from ..utils import LightkurveWarning

log = logging.getLogger(__name__)

__all__ = ['SeismologyButler']


class SeismologyButler(object):
    """Enables astroseismic quantities to be estimated from periodograms.

    This class provides easy access to methods to estimate numax, deltanu, radius,
    mass, and logg, and stores them on its tray for easy diagnostic plotting.
    """
    def __init__(self, periodogram):
        if not isinstance(periodogram, SNRPeriodogram):
            warnings.warn("SeismologyButler received a periodogram which does not "
                          "appear to have been background-corrected. Consider calling "
                          "`periodogram.flatten()` prior to extracting seismological parameters.",
                          LightkurveWarning)
        self.periodogram = periodogram

    def __repr__(self):
        attrs = np.asarray(['numax', 'deltanu', 'mass', 'radius', 'logg'])
        tray = np.asarray([hasattr(self, attr) for attr in attrs])
        if tray.sum() == 0:
            tray_str = '\n\t|  Tray is empty.  |\n\t ' + '-'*(18 + len(', '.join(attrs[tray])))
        else:
            tray_str = '\n\t|  On tray: ' + ', '.join(attrs[tray])+'  |\n\t '+'-'*(13 + len(', '.join(attrs[tray])))


        return 'SeismologyButler(ID: {})\n{}'.format(self.periodogram.targetid, tray_str)

    @staticmethod
    def from_lightcurve(lc, **kwargs):
        """Returns a `SeismologyButler` given a `~lightkurve.lightcurve.LightCurve` object."""
        log.info("Building a SeismologyButler object directly from a light curve "
                 "uses default periodogram parameters. For further tuneability, "
                 "create a periodogram object first, using `to_periodogram`.")
        return SeismologyButler(periodogram=lc.to_periodogram(**kwargs).flatten())

    def _validate_method(self, method, supported_methods):
        """Raises ValueError if a method is not supported."""
        method = method.lower()
        if method in supported_methods:
            return method
        raise ValueError("method {} is not supported; "
                         "must be one of {}".format(method, supported_methods))

    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError("You need to call `SeismologyButler"
                                     ".estimate_numax()` first.")
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError("You need to call `SeismologyButler"
                                     ".estimate_deltanu()` first.")
        return deltanu

    def plot_echelle(self, deltanu=None, numax=None,
                     minimum_frequency=None, maximum_frequency=None,
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
        scale: str
            Set z axis to be "linear" or "log". Default is linear.
        cmap : str
            The name of the matplotlib colourmap to use in the echelle diagram.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        freq = self.periodogram.frequency  # Makes code below more readable

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
            if fmin < 0.:
                fmin = 0.

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
        pp = self.periodogram.power[int(fmin/fs)-x0:int(fmax/fs)-x0] # Power range

        n_rows = int((ff[-1]-ff[0])/deltanu) # Number of stacks to use
        n_columns = int(deltanu/fs)          # Number of elements in each stack

        # Reshape the power into n_rows of n_columns
        ep = np.reshape(pp[:(n_rows*n_columns)], (n_rows, n_columns))

        if scale=='log':
            ep = np.log10(ep)

        #Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[:(n_rows*n_columns)],(n_rows,n_columns))
        x_f = ((ef[0,:]-ef[0,0]) % deltanu)
        y_f = (ef[:,0])

        #Plot the echelle diagram
        with plt.style.context(MPLSTYLE):
            fig, ax = plt.subplots()
            extent = (x_f[0].value, x_f[-1].value, y_f[0].value, y_f[-1].value)
            figsize = plt.rcParams['figure.figsize']
            a = figsize[1] / figsize[0]
            b = (extent[3] - extent[2]) / extent[1]
            vmin = np.nanpercentile(ep.value, 1)
            vmax = np.nanpercentile(ep.value, 99)
            im = ax.imshow(ep.value, cmap=cmap, aspect=a/b, origin='lower',
                      extent=extent, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax)
            if isinstance(self.periodogram, SNRPeriodogram):
                ylabel = 'Signal to Noise Ratio (SNR)'
            elif self.periodogram.power.unit == cds.ppm:
                ylabel = "Amplitude [{}]".format(self.periodogram.power.unit.to_string('latex'))
            else:
                ylabel = "Power Spectral Density [{}]".format(self.periodogram.power.unit.to_string('latex'))

            cbar.set_label(ylabel)
            ax.set_xlabel(r'Frequency mod. {:.2f}'.format(deltanu))
            ax.set_ylabel(r'Frequency [{}]'.format(freq.unit.to_string('latex')))
            ax.set_title('Echelle diagram for {}'.format(self.periodogram.label))

        return ax


    def estimate_numax(self, method="acf", **kwargs):
        """Returns the frequency of the peak of the seismic oscillation modes envelope.

        At present, the only method supported is based on using a
        2D autocorrelation function (ACF).  This method is implemented by the
        `~lightkurve.seismology.estimate_numax_acf` function which accepts
        the parameters `numaxs`, `window_width`, and `spacing`.
        For details and literature references, please read the detailed
        docstring of this function by typing ``lightkurve.seismology.estimate_numax_acf?``
        in a Python terminal or notebook.

        Parameters
        ----------
        method : str
            Method to use. Only ``"acf"`` is supported at this time.

        Returns
        -------
        numax : `~lightkurve.seismology.SeismologyQuantity`
            Numax of the periodogram, including details on the units and method.
        """
        method = self._validate_method(method, supported_methods=["acf"])
        if method == "acf":
            from .numax_estimators import estimate_numax_acf
            result = estimate_numax_acf(self.periodogram, **kwargs)
        self.numax = result
        return result

    def diagnose_numax(self, numax=None):
        """Create diagnostic plots showing how numax was estimated."""
        numax = self._validate_numax(numax)
        return numax.diagnostics_plot_method(numax, self.periodogram)

    def estimate_deltanu(self, method='acf', numax=None):
        """Returns the average value of the large frequency spacing, DeltaNu,
        of the seismic oscillations of the target.

        At present, the only method supported is based on using an
        autocorrelation function (ACF).  This method is implemented by the
        `~lightkurve.seismology.estimate_deltanu_acf` function which requires
        the parameter `numax`. For details and literature references, please
        read the detailed docstring of this function by typing
        ``lightkurve.seismology.estimate_deltanu_acf?`` in a Python terminal or notebook.

        Parameters
        ----------
        method : str
            Method to use. Only ``"acf"`` is supported at this time.

        Returns
        -------
        deltanu : `~lightkurve.seismology.SeismologyQuantity`
            DeltaNu of the periodogram, including details on the units and method.
        """
        method = self._validate_method(method, supported_methods=["acf"])
        numax = self._validate_numax(numax)

        if method == "acf":
            from .deltanu_estimators import estimate_deltanu_acf
            result = estimate_deltanu_acf(self.periodogram, numax=numax)

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
