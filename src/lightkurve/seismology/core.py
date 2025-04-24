"""Defines the Seismology class."""
import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from astropy import units as u
from astropy.units import cds

from .. import MPLSTYLE
from . import utils, stellar_estimators
from ..periodogram import SNRPeriodogram
from ..utils import LightkurveWarning, validate_method
from .utils import SeismologyQuantity

# Import the optional Bokeh dependency required by ``interact_echelle```,
# or print a friendly error otherwise.
_BOKEH_IMPORT_ERROR = None
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure
    from bokeh.models import LogColorMapper, Slider, RangeSlider, Button
    from bokeh.layouts import layout, Spacer
except Exception as e:
    # We will print a nice error message when ``interact_echelle``` is called.
    # the error would be raised there in case users need to diagnose problems
    _BOKEH_IMPORT_ERROR = e

log = logging.getLogger(__name__)

__all__ = ["Seismology"]


class Seismology(object):
    """Enables astroseismic quantities to be estimated from periodograms.

    This class provides easy access to methods to estimate numax, deltanu, radius,
    mass, and logg, and stores them on its tray for easy diagnostic plotting.

    Examples
    --------
    Download the TESS light curve for HIP 116158:

        >>> import lightkurve as lk
        >>> lc = lk.search_lightcurve("HIP 116158", sector=2).download()  # doctest: +SKIP
        >>> lc = lc.normalize().remove_nans().remove_outliers()  # doctest: +SKIP

    Create a Lomb-Scargle periodogram:

        >>> pg = lc.to_periodogram(normalization='psd', minimum_frequency=100, maximum_frequency=800)  # doctest: +SKIP

    Create a Seismology object and use it to estimate parameters:

        >>> seismology = pg.flatten().to_seismology()  # doctest: +SKIP
        >>> seismology.estimate_numax()  # doctest: +SKIP
        numax: 415.00 uHz (method: ACF2D)
        >>> seismology.estimate_deltanu()  # doctest: +SKIP
        deltanu: 28.78 uHz (method: ACF2D)
        >>> seismology.estimate_radius(teff=5080)  # doctest: +SKIP
        radius: 2.78 solRad (method: Uncorrected Scaling Relations)

    Parameters
    ----------
    periodogram : `~lightkurve.periodogram.Periodogram` object
        Periodogram to be analyzed. Must be background-corrected,
        e.g. using `periodogram.flatten()`.
    """

    periodogram = None
    """The periodogram from which seismological parameters are being extracted."""

    def __init__(self, periodogram):
        if not isinstance(periodogram, SNRPeriodogram):
            warnings.warn(
                "Seismology received a periodogram which does not appear "
                "to have been background-corrected. Please consider calling "
                "`periodogram.flatten()` prior to extracting seismological parameters.",
                LightkurveWarning,
            )
        self.periodogram = periodogram

    def __repr__(self):
        attrs = np.asarray(["numax", "deltanu", "mass", "radius", "logg"])
        tray = np.asarray([hasattr(self, attr) for attr in attrs])
        if tray.sum() == 0:
            tray_str = " - no values have been computed so far."
        else:
            tray_str = " - computed values:\n * " + "\n * ".join(
                [getattr(self, attr).__repr__() for attr in attrs[tray]]
            )
        return "Seismology(ID: {}){}".format(self.periodogram.label, tray_str)

    @staticmethod
    def from_lightcurve(lc, **kwargs):
        """Returns a `Seismology` object given a `LightCurve`."""
        log.info(
            "Building a Seismology object directly from a light curve "
            "uses default periodogram parameters. For further tuneability, "
            "create a periodogram object first, using `to_periodogram`."
        )
        return Seismology(
            periodogram=lc.normalize()
            .remove_nans()
            .fill_gaps()
            .to_periodogram(**kwargs)
            .flatten()
        )

    def _validate_numax(self, numax):
        """Raises exception if `numax` is None and `self.numax` is not set."""
        if numax is None:
            try:
                return self.numax
            except AttributeError:
                raise AttributeError(
                    "You need to call `Seismology.estimate_numax()` first."
                )
        return numax

    def _validate_deltanu(self, deltanu):
        """Raises exception if `deltanu` is None and `self.deltanu` is not set."""
        if deltanu is None:
            try:
                return self.deltanu
            except AttributeError:
                raise AttributeError(
                    "You need to call `Seismology.estimate_deltanu()` first."
                )
        return deltanu

    def _clean_echelle(
        self,
        deltanu=None,
        numax=None,
        minimum_frequency=None,
        maximum_frequency=None,
        smooth_filter_width=0.1,
        scale="linear",
    ):
        """Takes input seismology object and creates the necessary arrays for an echelle
        diagram. Validates all the inputs.

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

        Returns
        -------
        ep : np.ndarray
            Echelle diagram power
        x_f : np.ndarray
            frequencies for X axis
        y_f : np.ndarray
            frequencies for Y axis
        """
        if (minimum_frequency is None) & (maximum_frequency is None):
            numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)

        if (not hasattr(numax, "unit")) & (numax is not None):
            numax = numax * self.periodogram.frequency.unit
        if (not hasattr(deltanu, "unit")) & (deltanu is not None):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
            power = pgsmooth.power  # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable
            power = self.periodogram.power  # Makes code below more readable

        fmin = freq[0]
        fmax = freq[-1]

        # Check for any superfluous input
        if (numax is not None) & (
            any([a is not None for a in [minimum_frequency, maximum_frequency]])
        ):
            warnings.warn(
                "You have passed both a numax and a frequency limit. "
                "The frequency limit will override the numax input.",
                LightkurveWarning,
            )

        # Ensure input numax is in the correct units (if there is one)
        if numax is not None:
            numax = u.Quantity(numax, freq.unit).value
            if numax > freq[-1].value:
                raise ValueError(
                    "You can't pass in a numax outside the"
                    "frequency range of the periodogram."
                )

            fwhm = utils.get_fwhm(self.periodogram, numax)

            fmin = numax - 2 * fwhm
            if fmin < freq[0].value:
                fmin = freq[0].value

            fmax = numax + 2 * fwhm
            if fmax > freq[-1].value:
                fmax = freq[-1].value

        # Set limits and set them in the right units
        if minimum_frequency is not None:
            fmin = u.Quantity(minimum_frequency, freq.unit).value
            if fmin > freq[-1].value:
                raise ValueError(
                    "You can't pass in a limit outside the "
                    "frequency range of the periodogram."
                )

        if maximum_frequency is not None:
            fmax = u.Quantity(maximum_frequency, freq.unit).value
            if fmax > freq[-1].value:
                raise ValueError(
                    "You can't pass in a limit outside the "
                    "frequency range of the periodogram."
                )

        # Make sure fmin and fmax are Quantities or code below will break
        fmin = u.Quantity(fmin, freq.unit)
        fmax = u.Quantity(fmax, freq.unit)

        # Add on 1x deltanu so we don't miss off any important range due to rounding
        if fmax < freq[-1] - 1.5 * deltanu:
            fmax += deltanu

        fs = np.median(np.diff(freq))
        x0 = int(freq[0] / fs)

        ff = freq[int(fmin / fs) - x0 : int(fmax / fs) - x0]  # Selected frequency range
        pp = power[int(fmin / fs) - x0 : int(fmax / fs) - x0]  # Power range

        # Reshape the power into n_rows of n_columns
        #  When modulus ~ zero, deltanu divides into frequency without remainder
        mod_zeros = find_peaks(-1.0 * (ff % deltanu))[0]

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
        n_columns = int((approx_end - start) / n_rows)

        # The exact end point is therefore the ncolumns*nrows away from the start
        end = start + n_columns * n_rows

        ep = np.reshape(pp[start:end], (n_rows, n_columns))

        if scale == "log":
            ep = np.log10(ep)

        # Reshape the freq into n_rowss of n_columnss & create arays
        ef = np.reshape(ff[start:end], (n_rows, n_columns))
        x_f = (ef[0, :] - ef[0, 0]) % deltanu
        y_f = ef[:, 0]
        return ep, x_f, y_f

    def plot_echelle(
        self,
        deltanu=None,
        numax=None,
        minimum_frequency=None,
        maximum_frequency=None,
        smooth_filter_width=0.1,
        scale="linear",
        ax=None,
        cmap="Blues",
    ):
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
        scale: str
            Set z axis to be "linear" or "log". Default is linear.
        cmap : str
            The name of the matplotlib colourmap to use in the echelle diagram.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if (minimum_frequency is None) & (maximum_frequency is None):
            numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)

        if (not hasattr(numax, "unit")) & (numax is not None):
            numax = numax * self.periodogram.frequency.unit
        if (not hasattr(deltanu, "unit")) & (deltanu is not None):
            deltanu = deltanu * self.periodogram.frequency.unit

        ep, x_f, y_f = self._clean_echelle(
            numax=numax,
            deltanu=deltanu,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            smooth_filter_width=smooth_filter_width,
        )

        # Plot the echelle diagram
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            extent = (x_f[0].value, x_f[-1].value, y_f[0].value, y_f[-1].value)
            figsize = plt.rcParams["figure.figsize"]
            a = figsize[1] / figsize[0]
            b = (extent[3] - extent[2]) / (extent[1] - extent[0])
            vmin = np.nanpercentile(ep.value, 1)
            vmax = np.nanpercentile(ep.value, 99)

            im = ax.imshow(
                ep.value,
                cmap=cmap,
                aspect=a / b,
                origin="lower",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )

            cbar = plt.colorbar(im, ax=ax, extend="both", pad=0.01)

            if isinstance(self.periodogram, SNRPeriodogram):
                ylabel = "Signal to Noise Ratio (SNR)"
            elif self.periodogram.power.unit == cds.ppm:
                ylabel = "Amplitude [{}]".format(
                    self.periodogram.power.unit.to_string("latex")
                )
            else:
                ylabel = "Power Spectral Density [{}]".format(
                    self.periodogram.power.unit.to_string("latex")
                )

            if scale == "log":
                ylabel = "log10(" + ylabel + ")"

            cbar.set_label(ylabel)
            ax.set_xlabel(r"Frequency mod. {:.2f}".format(deltanu))
            ax.set_ylabel(
                r"Frequency [{}]".format(
                    self.periodogram.frequency.unit.to_string("latex")
                )
            )
            ax.set_title("Echelle diagram for {}".format(self.periodogram.label))

        return ax

    def _make_echelle_elements(
        self,
        deltanu,
        cmap="viridis",
        minimum_frequency=None,
        maximum_frequency=None,
        smooth_filter_width=0.1,
        scale="linear",
        width=490,
        height=340,
        title="Echelle",
    ):
        """Helper function to make the elements of the echelle diagram for bokeh plotting."""
        if not hasattr(deltanu, "unit"):
            deltanu = deltanu * self.periodogram.frequency.unit

        if smooth_filter_width:
            pgsmooth = self.periodogram.smooth(filter_width=smooth_filter_width)
            freq = pgsmooth.frequency  # Makes code below more readable below
        else:
            freq = self.periodogram.frequency  # Makes code below more readable

        ep, x_f, y_f = self._clean_echelle(
            deltanu=deltanu,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            smooth_filter_width=smooth_filter_width,
            scale=scale,
        )

        fig = figure(
            width=width,
            height=height,
            x_range=(0, 1),
            y_range=(y_f[0].value, y_f[-1].value),
            title=title,
            tools="pan,box_zoom,reset",
            toolbar_location="above",
            border_fill_color="white",
        )

        fig.yaxis.axis_label = r"Frequency [{}]".format(freq.unit.to_string())
        fig.xaxis.axis_label = r"Frequency / {:.3f} Mod. 1".format(deltanu)

        lo, hi = np.nanpercentile(ep.value, [0.1, 99.9])
        vlo, vhi = 0.3 * lo, 1.7 * hi
        vstep = (lo - hi) / 500
        color_mapper = LogColorMapper(palette="RdYlGn10", low=lo, high=hi)

        fig.image(
            image=[ep.value],
            x=0,
            y=y_f[0].value,
            dw=1,
            dh=y_f[-1].value,
            color_mapper=color_mapper,
            name="img",
        )

        stretch_slider = RangeSlider(
            start=vlo,
            end=vhi,
            step=vstep,
            title="",
            value=(lo, hi),
            orientation="vertical",
            width=10,
            height=230,
            direction="rtl",
            show_value=False,
            sizing_mode="fixed",
            name="stretch",
        )

        def stretch_change_callback(attr, old, new):
            """TPF stretch slider callback."""
            fig.select("img")[0].glyph.color_mapper.high = new[1]
            fig.select("img")[0].glyph.color_mapper.low = new[0]

        stretch_slider.on_change("value", stretch_change_callback)
        return fig, stretch_slider

    def interact_echelle(self, notebook_url="localhost:8888", **kwargs):
        """Display an interactive Jupyter notebook widget showing an Echelle diagram.

        This feature only works inside an active Jupyter Notebook, and
        requires an optional dependency, ``bokeh`` (v1.0 or later).
        This dependency can be installed using e.g. `conda install bokeh`.

        Parameters
        ----------
        notebook_url : str
            Location of the Jupyter notebook page (default: "localhost:8888")
            When showing Bokeh applications, the Bokeh server must be
            explicitly configured to allow connections originating from
            different URLs. This parameter defaults to the standard notebook
            host and port. If you are running on a different location, you
            will need to supply this value for the application to display
            properly. If no protocol is supplied in the URL, e.g. if it is
            of the form "localhost:8888", then "http" will be used.
        """
        if _BOKEH_IMPORT_ERROR is not None:
            log.error(
                "The interact_echelle() tool requires the `bokeh` package; "
                "you can install bokeh using e.g. `conda install bokeh`."
            )
            raise _BOKEH_IMPORT_ERROR

        maximum_frequency = kwargs.pop(
            "maximum_frequency", self.periodogram.frequency.max().value
        )
        minimum_frequency = kwargs.pop(
            "minimum_frequency", self.periodogram.frequency.min().value
        )

        if not hasattr(self, "deltanu"):
            dnu = SeismologyQuantity(
                quantity=self.periodogram.frequency.max() / 30,
                name="deltanu",
                method="echelle",
            )
        else:
            dnu = self.deltanu

        def create_interact_ui(doc):
            fig_tpf, stretch_slider = self._make_echelle_elements(
                dnu,
                maximum_frequency=maximum_frequency,
                minimum_frequency=minimum_frequency,
                **kwargs
            )
            maxdnu = self.periodogram.frequency.max().value / 5
            # Interactive slider widgets
            dnu_slider = Slider(
                start=0.01,
                end=maxdnu,
                value=dnu.value,
                step=0.01,
                title="Delta Nu",
                width=290,
            )
            r_button = Button(label=">", button_type="default", width=30)
            l_button = Button(label="<", button_type="default", width=30)
            rr_button = Button(label=">>", button_type="default", width=30)
            ll_button = Button(label="<<", button_type="default", width=30)

            def update(attr, old, new):
                """Callback to take action when dnu slider changes"""
                dnu = SeismologyQuantity(
                    quantity=dnu_slider.value * u.microhertz,
                    name="deltanu",
                    method="echelle",
                )
                ep, _, _ = self._clean_echelle(
                    deltanu=dnu,
                    minimum_frequency=minimum_frequency,
                    maximum_frequency=maximum_frequency,
                    **kwargs
                )
                fig_tpf.select("img")[0].data_source.data["image"] = [ep.value]
                fig_tpf.xaxis.axis_label = r"Frequency / {:.3f} Mod. 1".format(dnu)

            def go_right_by_one_small():
                """Step forward in time by a single cadence"""
                existing_value = dnu_slider.value
                if existing_value < maxdnu:
                    dnu_slider.value = existing_value + 0.002

            def go_left_by_one_small():
                """Step back in time by a single cadence"""
                existing_value = dnu_slider.value
                if existing_value > 0:
                    dnu_slider.value = existing_value - 0.002

            def go_right_by_one():
                """Step forward in time by a single cadence"""
                existing_value = dnu_slider.value
                if existing_value < maxdnu:
                    dnu_slider.value = existing_value + 0.01

            def go_left_by_one():
                """Step back in time by a single cadence"""
                existing_value = dnu_slider.value
                if existing_value > 0:
                    dnu_slider.value = existing_value - 0.01

            dnu_slider.on_change("value", update)
            r_button.on_click(go_right_by_one_small)
            l_button.on_click(go_left_by_one_small)
            rr_button.on_click(go_right_by_one)
            ll_button.on_click(go_left_by_one)

            widgets_and_figures = layout(
                [fig_tpf, [Spacer(height=20), stretch_slider]],
                [
                    ll_button,
                    Spacer(width=30),
                    l_button,
                    Spacer(width=25),
                    dnu_slider,
                    Spacer(width=30),
                    r_button,
                    Spacer(width=23),
                    rr_button,
                ],
            )
            doc.add_root(widgets_and_figures)

        output_notebook(verbose=False, hide_banner=True)
        return show(create_interact_ui, notebook_url=notebook_url)

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

    def estimate_deltanu(self, method="acf2d", numax=None):
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

    def estimate_radius(self, teff=None, numax=None, deltanu=None):
        """Returns a stellar radius estimate based on the scaling relations.

        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar radius as

        R = Rsol * (numax/numax_sol)(deltanu/deltanusol)^-2(Teff/Teffsol)^0.5

        where R is the radius and Teff is the effective temperature, and the suffix
        'sol' indicates a solar value. In this method we use the solar values for
        numax and deltanu as given in Huber et al. (2011) and for Teff as given in
        Prsa et al. (2016).

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        If no value of effective temperature is given, this function will check the
        meta data of the `Periodogram` object used to create the `Seismology` object.
        These data will often contain an effective tempearture from the Kepler Input
        Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
        or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
        catalogues are estimated using photometry, and so have large associated uncertainties
        (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
        temperature are often more precise.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.
        deltanu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. If not given an astropy unit, assumed to be in units of
            microhertz.
        teff : float
            The effective temperature of the star. In units of Kelvin.
        numax_err : float
            Error on numax. Assumed to be same units as numax
        deltanu_err : float
            Error on deltanu. Assumed to be same units as deltanu
        teff_err : float
            Error on Teff. Assumed to be same units as Teff.

        Returns
        -------
        radius : `~lightkurve.seismology.SeismologyQuantity`
            Stellar radius estimate.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        if teff is None:
            teff = self.periodogram.meta.get("TEFF")
            if teff is None:
                raise ValueError(
                    "You must provide an effective temperature argument (`teff`) to `estimate_radius`,"
                    "because the Periodogram object does not contain it in its meta data (i.e. `pg.meta['TEFF']` is missing"
                )
            else:
                log.info(
                    "Using value for effective temperature from the Kepler Input Catalogue."
                    "These temperatue values may sometimes differ significantly from modern estimates."
                )
                pass
        else:
            pass

        result = stellar_estimators.estimate_radius(numax, deltanu, teff)
        self.radius = result
        return result

    def estimate_mass(self, teff=None, numax=None, deltanu=None):
        """Calculates mass using the asteroseismic scaling relations.

        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar mass as

        M = Msol * (numax/numax_sol)^3(deltanu/deltanusol)^-4(Teff/Teffsol)^1.5

        where M is the mass and Teff is the effective temperature, and the suffix
        'sol' indicates a solar value. In this method we use the solar values for
        numax and deltanu as given in Huber et al. (2011) and for Teff as given in
        Prsa et al. (2016).

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        If no value of effective temperature is given, this function will check the
        meta data of the `Periodogram` object used to create the `Seismology` object.
        These data will often contain an effective tempearture from the Kepler Input
        Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
        or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
        catalogues are estimated using photometry, and so have large associated uncertainties
        (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
        temperature are often more precise.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.
        deltanu : float
            The frequency spacing between two consecutive overtones of equal radial
            degree. If not given an astropy unit, assumed to be in units of
            microhertz.
        teff : float
            The effective temperature of the star. In units of Kelvin.
        numax_err : float
            Error on numax. Assumed to be same units as numax
        deltanu_err : float
            Error on deltanu. Assumed to be same units as deltanu
        teff_err : float
            Error on Teff. Assumed to be same units as Teff.

        Returns
        -------
        mass : `~lightkurve.seismology.SeismologyQuantity`
            Stellar mass estimate.
        """
        numax = self._validate_numax(numax)
        deltanu = self._validate_deltanu(deltanu)
        if teff is None:
            teff = self.periodogram.meta.get("TEFF")
            if teff is None:
                raise ValueError(
                    "You must provide an effective temperature argument (`teff`) to `estimate_radius`,"
                    "because the Periodogram object does not contain it in its meta data (i.e. `pg.meta['TEFF']` is missing"
                )
            else:
                log.info(
                    "Using value for effective temperature from the Kepler Input Catalogue."
                    "These temperatue values may sometimes differ significantly from modern estimates."
                )
                pass
        else:
            pass

        result = stellar_estimators.estimate_mass(numax, deltanu, teff)
        self.mass = result
        return result

    def estimate_logg(self, teff=None, numax=None):
        """Calculates the log of the surface gravity using the asteroseismic scaling
        relations.

        The two global observable seismic parameters, numax and deltanu, along with
        temperature, scale with fundamental stellar properties (Brown et al. 1991;
        Kjeldsen & Bedding 1995). These scaling relations can be rearranged to
        calculate a stellar surface gravity as

            g = gsol * (numax/numax_sol)(Teff/Teffsol)^0.5

        where g is the surface gravity and Teff is the effective temperature,
        and the suffix 'sol' indicates a solar value. In this method we use the
        solar values for numax as given in Huber et al. (2011) and for Teff as given
        in Prsa et al. (2016). The solar surface gravity is calcluated from the
        astropy constants for solar mass and radius and does not have an error.

        The solar surface gravity is returned as log10(g) with units in dex, as is
        common in the astrophysics literature.

        This code structure borrows from work done in Bellinger et al. (2019), which
        also functions as an accessible explanation of seismic scaling relations.

        If no value of effective temperature is given, this function will check the
        meta data of the `Periodogram` object used to create the `Seismology` object.
        These data will often contain an effective tempearture from the Kepler Input
        Catalogue (KIC, https://ui.adsabs.harvard.edu/abs/2011AJ....142..112B/abstract),
        or from the EPIC or TIC for K2 and TESS respectively. The temperature values in these
        catalogues are estimated using photometry, and so have large associated uncertainties
        (roughly 200 K, see KIC). For more better results, spectroscopic measurements of
        temperature are often more precise.

        NOTE: These scaling relations are scaled to the Sun, and therefore do not
        always produce an entirely accurate result for more evolved stars.

        Parameters
        ----------
        numax : float
            The frequency of maximum power of the seismic mode envelope. If not
            given an astropy unit, assumed to be in units of microhertz.
        teff : float
            The effective temperature of the star. In units of Kelvin.
        numax_err : float
            Error on numax. Assumed to be same units as numax
        teff_err : float
            Error on teff. Assumed to be same units as teff.

        Returns
        -------
        logg : `~lightkurve.seismology.SeismologyQuantity`
            Stellar surface gravity estimate.
        """
        numax = self._validate_numax(numax)
        if teff is None:
            teff = self.periodogram.meta.get("TEFF")
            if teff is None:
                raise ValueError(
                    "You must provide an effective temperature argument (`teff`) to `estimate_radius`,"
                    "because the Periodogram object does not contain it in its meta data (i.e. `pg.meta['TEFF']` is missing"
                )
            else:
                log.info(
                    "Using value for effective temperature from the Kepler Input Catalogue."
                    "These temperatue values may sometimes differ significantly from modern estimates."
                )
                pass
        else:
            pass
        result = stellar_estimators.estimate_logg(numax, teff)
        self.logg = result
        return result
