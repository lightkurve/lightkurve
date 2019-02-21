"""This module provides helper functions for the `LightCurve.interact_bls()` feature."""
import logging
import warnings
import numpy as np
from astropy.convolution import convolve, Box1DKernel

log = logging.getLogger(__name__)

# Import the optional AstroPy dependency, or print a friendly error otherwise.
try:
    from astropy.stats.bls import BoxLeastSquares
except ImportError:
    pass  # we will print an error message in `show_interact_widget` instead

# Import the optional Bokeh dependency, or print a friendly error otherwise.
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import Selection, Slider, Span, Range1d
    from bokeh.models import Text
    from bokeh.layouts import layout, Spacer
    from bokeh.models.tools import HoverTool
    from bokeh.models.widgets import Button, Paragraph
    from bokeh.events import PanEnd, Reset
except ImportError:
    pass  # we will print an error message in `show_interact_widget` instead

from .interact import prepare_lightcurve_datasource
from .lightcurve import LightCurve


__all__ = ['show_interact_widget']


def prepare_bls_datasource(result, loc):
    """Prepare a bls result for bokeh plotting

    Parameters
    ----------
    result : BLS.model result
        The BLS model result to use
    loc : int
        Index of the "best" period. (Usually the max power)

    Returns
    -------
    bls_source : Bokeh.plotting.ColumnDataSource
        Bokeh style source for plotting
    """
    preselected = Selection()
    preselected.indices = [loc]
    bls_source = ColumnDataSource(data=dict(
                                        period=result['period'],
                                        power=result['power'],
                                        depth=result['depth'],
                                        duration=result['duration'],
                                        transit_time=result['transit_time']),
                                  selected=preselected)
    return bls_source


def prepare_folded_datasource(folded_lc):
    """Prepare a FoldedLightCurve object for bokeh plotting.

    Parameters
    ----------
    folded_lc : lightkurve.FoldedLightCurve
        The folded lightcurve

    Returns
    -------
    folded_source : Bokeh.plotting.ColumnDataSource
        Bokeh style source for plotting
    """
    folded_src = ColumnDataSource(data=dict(
                                  phase=np.sort(folded_lc.time),
                                  flux=folded_lc.flux[np.argsort(folded_lc.time)]))
    return folded_src


# Helper functions for help text...

def prepare_lc_help_source(lc):
    data = dict(time=[(np.max(lc.time) - np.min(lc.time)) * 0.98 + np.min(lc.time)],
                flux=[(np.max(lc.flux) - np.min(lc.flux)) * 0.9 + np.min(lc.flux)],
                boxicon=['https://bokeh.pydata.org/en/latest/_images/BoxZoom.png'],
                panicon=['https://bokeh.pydata.org/en/latest/_images/Pan.png'],
                reseticon=['https://bokeh.pydata.org/en/latest/_images/Reset.png'],
                tapicon=['https://bokeh.pydata.org/en/latest/_images/Tap.png'],
                hovericon=['https://bokeh.pydata.org/en/latest/_images/Hover.png'],
                helpme=['?'],
                help=["""
                             <div style="width: 550px;">
                                 <div>
                                     <span style="font-size: 12px; font-weight: bold;">Light Curve</span>
                                 </div>
                                 <div>
                                     <span style="font-size: 11px;"">This panel shows the full light curve, with the BLS model overlayed in red. The period of the model is the period
                                     currently selected in the BLS panel [top, left], indicated by the vertical red line. The duration of the transit model is given by the duration slider below..</span>
                                     <br></br>
                                 </div>
                                 <div>
                                     <span style="font-size: 12px; font-weight: bold;">Bokeh Tools</span>
                                 </div>
                                 <div>
                                     <span style="font-size: 11px;"">Each of the three panels have Bokeh tools to navigate them.
                                     You can turn off/on each tool by clicking the icon in the tray below each panel.
                                     You can zoom in using the Box Zoom Tool, move about the panel using the Pan Tool,
                                     or reset the panel back to the original view using the Reset Tool. </span>
                                     <br></br>
                                     <center>
                                         <table>
                                             <tr>
                                                 <td><img src="@boxicon" height="20" width="20"></td><td><span style="font-size: 11px;"">Box Zoom Tool</span></td>
                                             </tr>
                                             <tr>
                                                 <td><img src="@panicon" height="20" width="20"></td><td><span style="font-size: 11px;"">Pan Tool</span></td>
                                             </tr>
                                             <tr>
                                                 <td><img src="@reseticon" height="20" width="20"></td><td><span style="font-size: 11px;"">Reset Tool</span></td>
                                             </tr>
                                             <tr>
                                                 <td><img src="@tapicon" height="20" width="20"></td><td><span style="font-size: 11px;"">Tap Tool (select periods in BLS Panel only)</span></td>
                                             </tr>
                                             <tr>
                                                 <td><img src="@hovericon" height="20" width="20"></td><td><span style="font-size: 11px;"">Help Messages (click to disable/enable help)</span></td>
                                             </tr>
                                         </table>
                                     </center>
                                 </div>
                             </div>
                         """])
    return ColumnDataSource(data=data)


def prepare_bls_help_source(bls_source, slider_value):
    data = dict(period=[bls_source.data['period'][int(slider_value*0.95)]],
                power=[(np.max(bls_source.data['power']) - np.min(bls_source.data['power'])) * 0.98 + np.min(bls_source.data['power'])],
                helpme=['?'],
                help=["""
                             <div style="width: 375px;">
                                 <div style="height: 190px;">
                                 </div>
                                 <div>
                                     <span style="font-size: 12px; font-weight: bold;">Box Least Squares Periodogram</span>
                                 </div>
                                 <div>
                                     <span style="font-size: 11px;"">This panel shows the BLS periodogram for
                                      the light curve shown in the lower panel.
                                     The current selected period is highlighted by the red line.
                                     The selected period is the peak period within the range.
                                     The Folded Light Curve panel [right] will update when a new period
                                     is selected in the BLS Panel. You can select a new period either by
                                     using the Box Zoom tool to select a smaller range, or by clicking on the peak you want to select. </span>
                                     <br></br>
                                     <span style="font-size: 11px;"">The panel is set at the resolution
                                     given by the Resolution Slider [bottom]. This value is the number
                                     of points in the BLS Periodogram panel.
                                     Increasing the resolution will make the BLS Periodogram more accurate,
                                     but slower to render. To increase the resolution for a given peak,
                                     simply zoom in with the Box Zoom Tool.</span>

                                 </div>
                             </div>
                         """])
    return ColumnDataSource(data=data)


def prepare_f_help_source(f):
    data = dict(phase=[(np.max(f.time) - np.min(f.time)) * 0.98 + np.min(f.time)],
                flux=[(np.max(f.flux) - np.min(f.flux)) * 0.98 + np.min(f.flux)],
                helpme=['?'],
                help=["""
                        <div style="width: 375px;">
                            <div style="height: 190px;">
                            </div>
                            <div>
                                <span style="font-size: 12px; font-weight: bold;">Folded Light Curve</span>
                            </div>
                            <div>
                                <span style="font-size: 11px;"">This panel shows the folded light curve,
                                using the period currently selected in the BLS panel [left], indicated by the red line.
                                The transit model is show in red, and duration of the transit model
                                is given by the duration slider below. Update the slider to change the duration.
                                The period and transit midpoint values of the model are given above this panel.</span>
                                <br></br>
                                <span style="font-size: 11px;"">If the folded transit looks like a near miss of
                                the true period, try zooming in on the peak in the BLS Periodogram panel [right]
                                with the Box Zoom tool. This will increase the resolution of the peak, and provide
                                a better period solution. You can also vary the transit duration, for a better fit.
                                If the transit model is too shallow, it may be that you have selected a harmonic.
                                Look in the BLS Periodogram for a peak at (e.g. 0.25x, 0.5x, 2x, 4x the current period etc).</span>
                            </div>
                        </div>
                """])
    return ColumnDataSource(data=data)


def make_lightcurve_figure_elements(lc, model_lc, lc_source, model_lc_source, help_source):
    """Make a figure with a simple light curve scatter and model light curve line.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        Light curve to plot
    model_lc :  lightkurve.LightCurve
        Model light curve to plot
    lc_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for plotting light curve
    model_lc_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for plotting model light curve
    help_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for rendering help button

    Returns
    -------
    fig : bokeh.plotting.figure
        Bokeh figure object
    """
    # Make figure
    fig = figure(title='Light Curve', plot_height=300, plot_width=900,
                 tools="pan,box_zoom,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF", active_drag="box_zoom")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux (e/s)'
    if lc.time_format == 'bkjd':
        fig.xaxis.axis_label = 'Time - 2454833 (days)'
    elif lc.time_format == 'btjd':
        fig.xaxis.axis_label = 'Time - 2457000 (days)'
    else:
        fig.xaxis.axis_label = 'Time (days)'
    ylims = [np.nanmin(lc.flux), np.nanmax(lc.flux)]
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add light curve
    fig.circle('time', 'flux', line_width=1, color='#191919',
               source=lc_source, nonselection_line_color='#191919', size=0.5,
               nonselection_line_alpha=1.0)
    # Add model
    fig.step('time', 'flux', line_width=1, color='firebrick',
             source=model_lc_source, nonselection_line_color='firebrick',
             nonselection_line_alpha=1.0)

    # Help button
    question_mark = Text(x="time", y="flux", text="helpme", text_color="grey",
                         text_align='center', text_baseline="middle",
                         text_font_size='12px', text_font_style='bold',
                         text_alpha=0.6)
    fig.add_glyph(help_source, question_mark)
    help = fig.circle('time', 'flux', alpha=0.0, size=15, source=help_source,
                      line_width=2, line_color='grey', line_alpha=0.6)
    tooltips = help_source.data['help'][0]
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[help],
                            mode='mouse', point_policy="snap_to_data"))
    return fig


def make_folded_figure_elements(f, f_model_lc, f_source, f_model_lc_source, help_source):
    """Make a scatter plot of a FoldedLightCurve.

    Parameters
    ----------
    f : lightkurve.LightCurve
        Folded light curve to plot
    f_model_lc :  lightkurve.LightCurve
        Model folded light curve to plot
    f_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for plotting folded light curve
    f_model_lc_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for plotting model folded light curve
    help_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for rendering help button

    Returns
    -------
    fig : bokeh.plotting.figure
        Bokeh figure object
    """

    # Build Figure
    fig = figure(title='Folded Light Curve', plot_height=340, plot_width=450,
                 tools="pan,box_zoom,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF", active_drag="box_zoom")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux'
    fig.xaxis.axis_label = 'Phase'

    # Scatter point for data
    fig.circle('phase', 'flux', line_width=1, color='#191919',
               source=f_source, nonselection_line_color='#191919',
               nonselection_line_alpha=1.0, size=0.1)

    # Line plot for model
    fig.step('phase', 'flux', line_width=3, color='firebrick',
             source=f_model_lc_source, nonselection_line_color='firebrick',
             nonselection_line_alpha=1.0)

    # Help button
    question_mark = Text(x="phase", y="flux", text="helpme", text_color="grey",
                         text_align='center', text_baseline="middle",
                         text_font_size='12px', text_font_style='bold',
                         text_alpha=0.6)
    fig.add_glyph(help_source, question_mark)
    help = fig.circle('phase', 'flux', alpha=0.0, size=15, source=help_source,
                      line_width=2, line_color='grey', line_alpha=0.6)

    tooltips = help_source.data['help'][0]
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[help],
                            mode='mouse', point_policy="snap_to_data"))
    return fig


def make_bls_figure_elements(result, bls_source, help_source):
    """Make a line plot of a BLS result.

    Parameters
    ----------
    result : BLS.model result
        BLS model result to plot
    bls_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for plotting BLS source
    help_source : bokeh.plotting.ColumnDataSource
        Bokeh style source object for rendering help button

    Returns
    -------
    fig : bokeh.plotting.figure
        Bokeh figure object
    vertical_line : bokeh.models.Span
        Vertical line to highlight current selected period
    """

    # Build Figure
    fig = figure(title='BLS Periodogram', plot_height=340, plot_width=450,
                 tools="pan,box_zoom,tap,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF", x_axis_type='log', active_drag="box_zoom")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Power'
    fig.xaxis.axis_label = 'Period [days]'
    fig.y_range = Range1d(start=result.power.min() * 0.95, end=result.power.max() * 1.05)
    fig.x_range = Range1d(start=result.period.min(), end=result.period.max())

    # Add circles for the selection of new period. These are always hidden
    circ = fig.circle('period', 'power', source=bls_source, fill_alpha=0., size=6,
                      line_color=None, selection_color="white",
                      nonselection_fill_alpha=0.0,
                      nonselection_fill_color='white',
                      nonselection_line_color=None,
                      nonselection_line_alpha=0.0,
                      fill_color=None, hover_fill_color="white",
                      hover_alpha=0., hover_line_color="white")

    # Add line for the BLS power
    fig.line('period', 'power', line_width=1, color='#191919',
             source=bls_source, nonselection_line_color='#191919',
             nonselection_line_alpha=1.0)

    # Vertical line to indicate the current period
    vertical_line = Span(location=0, dimension='height',
                         line_color='firebrick', line_width=3, line_alpha=0.5)
    fig.add_layout(vertical_line)

    # Help button
    question_mark = Text(x="period", y="power", text="helpme", text_color="grey",
                         text_align='center', text_baseline="middle",
                         text_font_size='12px', text_font_style='bold',
                         text_alpha=0.6)
    fig.add_glyph(help_source, question_mark)
    help = fig.circle('period', 'power', alpha=0.0, size=15, source=help_source,
                      line_width=2, line_color='grey', line_alpha=0.6)
    tooltips = help_source.data['help'][0]
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[help],
                            mode='mouse', point_policy="snap_to_data"))

    return fig, vertical_line


def show_interact_widget(lc, notebook_url='localhost:8888', minimum_period=None,
                         maximum_period=None, resolution=2000):
    """Show the BLS interact widget.

    Parameters
    ----------
    notebook_url: str
        Location of the Jupyter notebook page (default: "localhost:8888")
        When showing Bokeh applications, the Bokeh server must be
        explicitly configured to allow connections originating from
        different URLs. This parameter defaults to the standard notebook
        host and port. If you are running on a different location, you
        will need to supply this value for the application to display
        properly. If no protocol is supplied in the URL, e.g. if it is
        of the form "localhost:8888", then "http" will be used.
    minimum_period : float or None
        Minimum period to assess the BLS to. If None, default value of 0.3 days
        will be used.
    maximum_period : float or None
        Maximum period to evaluate the BLS to. If None, the time coverage of the
        lightcurve / 4 will be used.
    resolution : int
        Number of points to use in the BLS panel. Lower this value to have a faster
        but less accurate compute time. You can also vary this value using the
        Resolution Slider.
    """
    try:
        import bokeh
        if bokeh.__version__[0] == '0':
            warnings.warn("interact_bls() requires Bokeh version 1.0 or later", LightkurveWarning)
    except ImportError:
        log.error("The interact_bls() tool requires the `bokeh` package; "
                  "you can install bokeh using e.g. `conda install bokeh`.")
        return None

    try:
        from astropy.stats.bls import BoxLeastSquares
    except ImportError:
        log.error("The `interact_bls()` tool requires the `astropy.stats.bls` module; "
                  "this requires AstroPy v3.1 or later.")

    def _create_interact_ui(doc, minp=minimum_period, maxp=maximum_period, resolution=resolution):
        """Create BLS interact user interface."""
        if minp is None:
            minp = 0.3
        if maxp is None:
            maxp = (lc.time[-1] - lc.time[0])/2

        time_format = ''
        if lc.time_format == 'bkjd':
            time_format = ' - 2454833 days'
        if lc.time_format == 'btjd':
            time_format = ' - 2457000 days'

        # Some sliders
        duration_slider = Slider(start=0.01,
                                 end=0.5,
                                 value=0.05,
                                 step=0.01,
                                 title="Duration [Days]",
                                 width=400)

        npoints_slider = Slider(start=500,
                                end=10000,
                                value=resolution,
                                step=100,
                                title="BLS Resolution",
                                width=400)

        # Set up the period values, BLS model and best period
        period_values = np.logspace(np.log10(minp), np.log10(maxp), npoints_slider.value)
        period_values = period_values[(period_values > duration_slider.value) &
                                        (period_values < maxp)]
        model = BoxLeastSquares(lc.time, lc.flux)
        result = model.power(period_values, duration_slider.value)
        loc = np.argmax(result.power)
        best_period = result.period[loc]
        best_t0 = result.transit_time[loc]

        # Some Buttons
        double_button = Button(label="Double Period", button_type="danger", width=100)
        half_button = Button(label="Half Period", button_type="danger", width=100)
        text_output = Paragraph(text="Period: {} days, T0: {}{}".format(
                                                    np.round(best_period, 7),
                                                    np.round(best_t0, 7), time_format),
                                width=350, height=40)

        # Set up BLS source
        bls_source = prepare_bls_datasource(result, loc)
        bls_help_source = prepare_bls_help_source(bls_source, npoints_slider.value)

        # Set up the model LC
        mf = model.model(lc.time, best_period, duration_slider.value, best_t0)
        mf /= np.median(mf)
        mask = ~(convolve(np.asarray(mf == np.median(mf)), Box1DKernel(2)) > 0.9)
        model_lc = LightCurve(lc.time[mask], mf[mask])
        model_lc = model_lc.append(LightCurve([(lc.time[0] - best_t0) + best_period/2], [1]))
        model_lc = model_lc.append(LightCurve([(lc.time[0] - best_t0) + 3*best_period/2], [1]))

        model_lc_source = ColumnDataSource(data=dict(
                                     time=np.sort(model_lc.time),
                                     flux=model_lc.flux[np.argsort(model_lc.time)]))

        # Set up the LC
        nb = int(np.ceil(len(lc.flux)/5000))
        lc_source = prepare_lightcurve_datasource(lc[::nb])
        lc_help_source = prepare_lc_help_source(lc)

        # Set up folded LC
        nb = int(np.ceil(len(lc.flux)/10000))
        f = lc.fold(best_period, best_t0)
        f_source = prepare_folded_datasource(f[::nb])
        f_help_source = prepare_f_help_source(f)

        f_model_lc = model_lc.fold(best_period, best_t0)
        f_model_lc = LightCurve([-0.5], [1]).append(f_model_lc)
        f_model_lc = f_model_lc.append(LightCurve([0.5], [1]))

        f_model_lc_source = ColumnDataSource(data=dict(
                                 phase=f_model_lc.time,
                                 flux=f_model_lc.flux))

        def _update_light_curve_plot(event):
            """If we zoom in on LC plot, update the binning."""
            mint, maxt = fig_lc.x_range.start, fig_lc.x_range.end
            inwindow = (lc.time > mint) & (lc.time < maxt)
            nb = int(np.ceil(inwindow.sum()/5000))
            temp_lc = lc[inwindow]
            lc_source.data = {'time': temp_lc.time[::nb],
                              'flux': temp_lc.flux[::nb]}

        def _update_folded_plot(event):
            loc = np.argmax(bls_source.data['power'])
            best_period = bls_source.data['period'][loc]
            best_t0 = bls_source.data['transit_time'][loc]
            # Otherwise, we can just update the best_period index
            minphase, maxphase = fig_folded.x_range.start, fig_folded.x_range.end
            f = lc.fold(best_period, best_t0)
            inwindow = (f.time > minphase) & (f.time < maxphase)
            nb = int(np.ceil(inwindow.sum()/10000))
            f_source.data = {'phase': f[inwindow].time[::nb],
                             'flux': f[inwindow].flux[::nb]}

        # Function to update the widget
        def _update_params(all=False, best_period=None, best_t0=None):
            if all:
                # If we're updating everything, recalculate the BLS model
                minp, maxp = fig_bls.x_range.start, fig_bls.x_range.end
                period_values = np.logspace(np.log10(minp), np.log10(maxp), npoints_slider.value)
                ok = (period_values > duration_slider.value) & (period_values < maxp)
                if ok.sum() == 0:
                    return
                period_values = period_values[ok]
                result = model.power(period_values, duration_slider.value)
                ok = np.isfinite(result['power']) & np.isfinite(result['duration']) &\
                         np.isfinite(result['transit_time']) & np.isfinite(result['period'])
                bls_source.data = dict(
                                     period=result['period'][ok],
                                     power=result['power'][ok],
                                     duration=result['duration'][ok],
                                     transit_time=result['transit_time'][ok])
                loc = np.nanargmax(bls_source.data['power'])
                best_period = bls_source.data['period'][loc]
                best_t0 = bls_source.data['transit_time'][loc]

                minpow, maxpow = bls_source.data['power'].min()*0.95,  bls_source.data['power'].max()*1.05
                fig_bls.y_range.start = minpow
                fig_bls.y_range.end = maxpow

            # Otherwise, we can just update the best_period index
            minphase, maxphase = fig_folded.x_range.start, fig_folded.x_range.end
            f = lc.fold(best_period, best_t0)
            inwindow = (f.time > minphase) & (f.time < maxphase)
            nb = int(np.ceil(inwindow.sum()/10000))
            f_source.data = {'phase': f[inwindow].time[::nb],
                             'flux': f[inwindow].flux[::nb]}

            mf = model.model(lc.time, best_period, duration_slider.value, best_t0)
            mf /= np.median(mf)
            mask = ~(convolve(np.asarray(mf == np.median(mf)), Box1DKernel(2)) > 0.9)
            model_lc = LightCurve(lc.time[mask], mf[mask])

            model_lc_source.data = {'time': np.sort(model_lc.time),
                                    'flux': model_lc.flux[np.argsort(model_lc.time)]}

            f_model_lc = model_lc.fold(best_period, best_t0)
            f_model_lc = LightCurve([-0.5], [1]).append(f_model_lc)
            f_model_lc = f_model_lc.append(LightCurve([0.5], [1]))

            f_model_lc_source.data = {'phase': f_model_lc.time,
                                      'flux': f_model_lc.flux}

            vertical_line.update(location=best_period)
            fig_folded.title.text = 'Period: {} days \t T0: {}{}'.format(
                                        np.round(best_period, 7),
                                        np.round(best_t0, 7), time_format)
            text_output.text = "Period: {} days, \t T0: {}{}".format(
                                        np.round(best_period, 7),
                                        np.round(best_t0, 7), time_format)

        # Callbacks
        def _update_upon_period_selection(attr, old, new):
            """When we select a period we should just update a few things, but we should not recalculate model
            """
            if len(new) > 0:
                new = new[0]
                best_period = bls_source.data['period'][new]
                best_t0 = bls_source.data['transit_time'][new]
                _update_params(best_period=best_period, best_t0=best_t0)

        def _update_model_slider(attr, old, new):
            """If the duration slider is updated, then update the whole model set."""
            _update_params(all=True)

        def _update_model_slider_EVENT(event):
            """If we update the duration slider, we should update the whole model set.
            This is the same as the _update_model_slider but it has a different call signature...
            """
            _update_params(all=True)

        def _double_period_event():
            fig_bls.x_range.start *= 2
            fig_bls.x_range.end *= 2
            _update_params(all=True)

        def _half_period_event():
            fig_bls.x_range.start /= 2
            fig_bls.x_range.end /= 2
            _update_params(all=True)

        # Help Hover Call Backs
        def _update_folded_plot_help_reset(event):
            f_help_source.data['phase'] = [(np.max(f.time) - np.min(f.time)) * 0.98 + np.min(f.time)]
            f_help_source.data['flux'] = [(np.max(f.flux) - np.min(f.flux)) * 0.98 + np.min(f.flux)]

        def _update_folded_plot_help(event):
            f_help_source.data['phase'] = [(fig_folded.x_range.end - fig_folded.x_range.start) * 0.95 + fig_folded.x_range.start]
            f_help_source.data['flux'] = [(fig_folded.y_range.end - fig_folded.y_range.start) * 0.95 + fig_folded.y_range.start]

        def _update_lc_plot_help_reset(event):
            lc_help_source.data['time'] = [(np.max(lc.time) - np.min(lc.time)) * 0.98 + np.min(lc.time)]
            lc_help_source.data['flux'] = [(np.max(lc.flux) - np.min(lc.flux)) * 0.9 + np.min(lc.flux)]

        def _update_lc_plot_help(event):
            lc_help_source.data['time'] = [(fig_lc.x_range.end - fig_lc.x_range.start) * 0.95 + fig_lc.x_range.start]
            lc_help_source.data['flux'] = [(fig_lc.y_range.end - fig_lc.y_range.start) * 0.9 + fig_lc.y_range.start]

        def _update_bls_plot_help_event(event):
            bls_help_source.data['period'] = [bls_source.data['period'][int(npoints_slider.value*0.95)]]
            bls_help_source.data['power'] = [(np.max(bls_source.data['power']) - np.min(bls_source.data['power'])) * 0.98
                                             + np.min(bls_source.data['power'])]

        def _update_bls_plot_help(attr, old, new):
            bls_help_source.data['period'] = [bls_source.data['period'][int(npoints_slider.value*0.95)]]
            bls_help_source.data['power'] = [(np.max(bls_source.data['power']) - np.min(bls_source.data['power'])) * 0.98
                                             + np.min(bls_source.data['power'])]

        # Create all the figures.
        fig_folded = make_folded_figure_elements(f, f_model_lc, f_source, f_model_lc_source, f_help_source)
        fig_folded.title.text = 'Period: {} days \t T0: {}{}'.format(np.round(best_period, 7), np.round(best_t0, 5), time_format)
        fig_bls, vertical_line = make_bls_figure_elements(result, bls_source, bls_help_source)
        fig_lc = make_lightcurve_figure_elements(lc, model_lc, lc_source, model_lc_source, lc_help_source)

        # Map changes

        # If we click a new period, update
        bls_source.selected.on_change('indices', _update_upon_period_selection)

        # If we change the duration, update everything, including help button for BLS
        duration_slider.on_change('value', _update_model_slider)
        duration_slider.on_change('value', _update_bls_plot_help)

        # If we increase resolution, update everything
        npoints_slider.on_change('value', _update_model_slider)

        # Make sure the vertical line always goes to the best period.
        vertical_line.update(location=best_period)

        # If we pan in the BLS panel, update everything
        fig_bls.on_event(PanEnd, _update_model_slider_EVENT)
        fig_bls.on_event(Reset, _update_model_slider_EVENT)

        # If we pan in the LC panel, rebin the points
        fig_lc.on_event(PanEnd, _update_light_curve_plot)
        fig_lc.on_event(Reset, _update_light_curve_plot)

        # If we pan in the Folded panel, rebin the points
        fig_folded.on_event(PanEnd, _update_folded_plot)
        fig_folded.on_event(Reset, _update_folded_plot)

        # Deal with help button
        fig_bls.on_event(PanEnd, _update_bls_plot_help_event)
        fig_bls.on_event(Reset, _update_bls_plot_help_event)
        fig_folded.on_event(PanEnd, _update_folded_plot_help)
        fig_folded.on_event(Reset, _update_folded_plot_help_reset)
        fig_lc.on_event(PanEnd, _update_lc_plot_help)
        fig_lc.on_event(Reset, _update_lc_plot_help_reset)

        # Buttons
        double_button.on_click(_double_period_event)
        half_button.on_click(_half_period_event)

        # Layout the widget
        doc.add_root(layout([
                            [fig_bls, fig_folded],
                            fig_lc,
                            [Spacer(width=70), duration_slider, Spacer(width=50), npoints_slider],
                            [Spacer(width=70), double_button, Spacer(width=70), half_button, Spacer(width=300), text_output]
                                ]))

    output_notebook(verbose=False, hide_banner=True)
    return show(_create_interact_ui, notebook_url=notebook_url)
