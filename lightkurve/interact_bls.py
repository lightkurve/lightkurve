"""Provides tools for bls interact

Example use
-----------
"""
import logging
import warnings
import numpy as np
from astropy.stats import sigma_clip
from .utils import KeplerQualityFlags, LightkurveWarning
import os


from astropy.stats.bls import BoxLeastSquares
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import time


log = logging.getLogger(__name__)

# Import the optional Bokeh dependency, or print a friendly error otherwise.
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import LogColorMapper, Selection, Slider, RangeSlider, \
        Span, ColorBar, LogTicker, Range1d
    from bokeh.models import Text
    from bokeh.layouts import layout, Spacer
    from bokeh.models.tools import HoverTool
    from bokeh.models.widgets import Button, Div
    from bokeh.models.formatters import PrintfTickFormatter
    from bokeh.events import PanEnd, Reset

    from bokeh.layouts import widgetbox
    from bokeh.models.widgets import Dropdown
except ImportError:
    log.critical('Bokeh is not installed. Interactive tools will not work.')


from .interact import prepare_lightcurve_datasource
from .lightcurve import LightCurve

from . import PACKAGEDIR


def prepare_bls_datasource(result, loc):
    '''Prepare a bls result for bokeh plotting
    '''
    preselected = Selection()
    preselected.indices = [loc]
    bls_source = ColumnDataSource(data=dict(
                                 period=result['period'],
                                 power=result['power'],
                                 depth=result['depth'],
                                 duration=result['duration'],
                                 transit_time=result['transit_time']), selected=preselected)
    return bls_source

def prepare_folded_datasource(f):
    '''Prepare a folded lightkurve.lightcurve for bokeh plotting
    '''
    folded_source = ColumnDataSource(data=dict(
                                 phase=f.time,
                                 flux=f.flux))
    return folded_source


def make_lightcurve_figure_elements(lc, model_lc, lc_source, model_lc_source):
    '''Make a figure with a simple light curve scatter and model light curve line
    '''
    fig = figure(title='Light Curve', plot_height=250, plot_width=900,
                 tools="pan,wheel_zoom,box_zoom,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux (e/s)'
    fig.xaxis.axis_label = 'Time - 2454833 (days)'

    ylims = [np.nanmin(lc.flux), np.nanmax(lc.flux)]
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add step lines, circles, and hover-over tooltips
    fig.circle('time', 'flux', line_width=1, color='#191919',
             source=lc_source, nonselection_line_color='#191919', size=2,
             nonselection_line_alpha=1.0)


    fig.step('time', 'flux', line_width=1, color='red',
             source=model_lc_source, nonselection_line_color='red',
             nonselection_line_alpha=1.0)



    circ = fig.circle('time', 'flux', source=lc_source, fill_alpha=0.3, size=8,
                      line_color=None, selection_color="firebrick",
                      nonselection_fill_alpha=0.0,
                      nonselection_fill_color='#191919',
                      nonselection_line_color=None,
                      nonselection_line_alpha=0.0,
                      fill_color=None, hover_fill_color="firebrick",
                      hover_alpha=0.9, hover_line_color="white")
    return fig


def make_folded_figure_elements(f, f_source, f_model_lc, f_model_lc_source, help_source):
    '''Make a scatter plot of a folded lightkurve.lightcurve
    '''
    fig = figure(title='Folded Light Curve', plot_height=340, plot_width=450,
                 tools="pan,wheel_zoom,box_zoom,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux'
    fig.xaxis.axis_label = 'Phase'
    fig.circle('phase', 'flux', line_width=1, color='#191919',
         source=f_source, nonselection_line_color='#191919',
         nonselection_line_alpha=1.0, size=2)
    fig.line('phase', 'flux', line_width=3, color='firebrick',
             source=f_model_lc_source, nonselection_line_color='firebrick',
             nonselection_line_alpha=1.0)

    help = fig.asterisk('phase', 'flux', alpha=0.5, size=10, source=help_source)
    tooltips = [("", "@help")]
    tooltips = """
                <div style="width: 350px;">
                    <div style="height: 130px;">
                    </div>
                    <div>
                        <span style="font-size: 12px; font-weight: bold;">Folded Light Curve</span>
                    </div>
                    <div>
                        <span style="font-size: 11px;"">This pane shows the folded light curve, using the period currently selected in the BLS pane [right], indicated by the red line.
                        You can zoom in on the transit at the center of the pane using the Box Zoom Tool or the Wheel Zoom Tool. You can also move about the
                        pane using the Pan Tool. You can reset the pane back to the original view using the Reset Tool.</span>
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
                                    <td><img src="@wheelicon" height="20" width="20"></td><td><span style="font-size: 11px;"">Wheel Zoom Tool</span></td>
                                </tr>
                                <tr>
                                    <td><img src="@reseticon" height="20" width="20"></td><td><span style="font-size: 11px;"">Reset Tool</span></td>
                                </tr>
                            </table>
                        </center>
                    </div>
                </div>
            """
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[help],
                            mode='mouse', point_policy="snap_to_data"))
    return fig


def make_bls_figure_elements(result, bls_source):
    ''' Make a line plot of a BLS result
    '''
    fig = figure(title='BLS (Select Period Values with Click or Box Zoom)', plot_height=340, plot_width=450,
                 tools="box_zoom,tap,reset",
                 toolbar_location="below",
                 border_fill_color="#FFFFFF", x_axis_type='log')
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Power'
    fig.xaxis.axis_label = 'Period [days]'
    fig.y_range = Range1d(start=result.power.min() * 0.95, end=result.power.max() * 1.05)
    fig.x_range = Range1d(start=result.period.min(), end=result.period.max())


    # Add step lines, circles, and hover-over tooltips
    fig.line('period', 'power', line_width=1, color='#191919',
             source=bls_source, nonselection_line_color='#191919',
             nonselection_line_alpha=1.0)
    circ = fig.circle('period', 'power', source=bls_source, fill_alpha=0.3, size=6,
                      line_color=None, selection_color="firebrick",
                      nonselection_fill_alpha=0.0,
                      nonselection_fill_color='#191919',
                      nonselection_line_color=None,
                      nonselection_line_alpha=0.0,
                      fill_color=None, hover_fill_color="firebrick",
                      hover_alpha=0.9, hover_line_color="white")

# Add period value hover tooltip
#    tooltips = [("Period", "@period")]
#    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[circ],
#                            mode='mouse', point_policy="snap_to_data"))

    # Vertical line to indicate the current period
    vertical_line = Span(location=0, dimension='height',
                         line_color='firebrick', line_width=3, line_alpha=0.5)
    offhand_lines = [Span(location=0, dimension='height',
                         line_color='blue', line_width=2, line_alpha=0.3, line_dash='dashed') for count in range(6)]
    fig.add_layout(vertical_line)
    [fig.add_layout(line) for line in offhand_lines]
    return fig, vertical_line, offhand_lines





def show_interact_widget(lc, notebook_url='localhost:8888'):
    ''' Show the widget
    '''

    def create_interact_ui(doc, minp=None, maxp=None):
        '''Create BLS interact user interface
        '''
        duration_slider = Slider(start=0.01,
                            end=0.3,
                            value=0.05,
                            step=0.01,
                            title="Duration [Days]",
                            width=400)

        npoints_slider = Slider(start=500,
                            end=10000,
                            value=2000,
                            step=100,
                            title="BLS Resolution [Points in BLS Plot]",
                            width=400)
        if minp is None:
            minp = np.mean(np.diff(lc.time)) * 20
        if maxp is None:
            maxp = (lc.time[-1] - lc.time[0])/4

        # Set up the period values, BLS model and best period
        period_values = np.logspace(np.log10(minp), np.log10(maxp), npoints_slider.value)
        model = BoxLeastSquares(lc.time, lc.flux)
        result = model.power(period_values, duration_slider.value)
        loc = np.argmax(result.power)
        best_period = result.period[loc]
        best_t0 = result.transit_time[loc]

        # Set up BLS source
        bls_source = prepare_bls_datasource(result, loc)

        # Set up the model LC
        model_lc = LightCurve(lc.time, model.model(lc.time, best_period, duration_slider.value, best_t0))
        model_lc_source = ColumnDataSource(data=dict(
                                     time=model_lc.time,
                                     flux=model_lc.flux))

        # Set up the LC
        lc_source = prepare_lightcurve_datasource(lc)

        # Set up folded LC
        f = lc.fold(best_period, best_t0)
        f_model_lc = model_lc.fold(best_period, best_t0)
        f_model_lc_source = ColumnDataSource(data=dict(
                                 phase=f_model_lc.time,
                                 flux=f_model_lc.flux))
        f_source = prepare_folded_datasource(f)
        f_help_source = ColumnDataSource(data=dict(
                                                 phase=[(np.max(f.time) - np.min(f.time)) * 0.98 + np.min(f.time)],
                                                 flux=[(np.max(f.flux) - np.min(f.flux)) * 0.98 + np.min(f.flux)],
                                                 boxicon=['https://bokeh.pydata.org/en/latest/_images/BoxZoom.png'],
                                                 panicon=['https://bokeh.pydata.org/en/latest/_images/Pan.png'],
                                                 wheelicon=['https://bokeh.pydata.org/en/latest/_images/WheelZoom.png'],
                                                 reseticon=['https://bokeh.pydata.org/en/latest/_images/Reset.png']
                                                 ))

        def update_params(all=False):
            if all:
                minp, maxp = fig_bls.x_range.start, fig_bls.x_range.end
                period_values = np.logspace(np.log10(minp), np.log10(maxp), npoints_slider.value)
                result = model.power(period_values, duration_slider.value)

                bls_source.data = dict(
                                     period=result['period'],
                                     power=result['power'],
                                     duration=result['duration'],
                                     transit_time=result['transit_time'])
                loc = np.argmax(bls_source.data['power'])
                best_period = bls_source.data['period'][loc]
                best_t0 = bls_source.data['transit_time'][loc]

                minpow, maxpow = result.power.min()*0.95, result.power.max()*1.05
                fig_bls.y_range.start = minpow
                fig_bls.y_range.end = maxpow

            f = lc.fold(best_period, best_t0)
            f_source.data['flux'] = f.flux
            f_source.data['phase'] = f.time

            model_lc = LightCurve(lc.time, model.model(lc.time, best_period, duration_slider.value, best_t0))
            model_lc_source.data['flux'] = model_lc.flux
            model_lc_source.data['time'] = model_lc.time
            f_model_lc = model_lc.fold(best_period, best_t0)
            f_model_lc_source.data['flux'] = f_model_lc.flux
            f_model_lc_source.data['phase'] = f_model_lc.time

            vertical_line.update(location=best_period)
            for idx, line in enumerate(offhand_lines[0:3]):
                line.update(location=best_period/(idx + 2))
            for idx, line in enumerate(offhand_lines[3:]):
                line.update(location=best_period*(idx + 2))
            fig_folded.title.text = 'Period: {} days \t T0: {}'.format(np.round(best_period, 7), np.round(best_t0, 5))

        # Callbacks
        def update_upon_period_selection(attr, old, new):
            ''' When we select a period we should just update a few things, but we should not recalculate model
            '''
            new = new[0]
            best_period = bls_source.data['period'][new]
            best_t0 = bls_source.data['transit_time'][new]
            update_params()


        def update_model_slider(attr, old, new):
            ''' If we update the duration slider, we should update the whole model set.
            '''
            update_params(all=True)

        def update_model_slider_EVENT(event):
            ''' If we update the duration slider, we should update the whole model set.
            This is the same as the update_model_slider but it has a different call signature...
            '''
            update_params(all=True)

        def update_folded_plot_reset(event):
            f_help_source.data['phase'] = [(np.max(f.time) - np.min(f.time)) * 0.98 + np.min(f.time)]
            f_help_source.data['flux'] = [(np.max(f.flux) - np.min(f.flux)) * 0.98 + np.min(f.flux)]

        def update_folded_plot(event):
            f_help_source.data['phase'] = [(fig_folded.x_range.end - fig_folded.x_range.start) * 0.95 + fig_folded.x_range.start]
            f_help_source.data['flux'] = [(fig_folded.y_range.end - fig_folded.y_range.start) * 0.95 + fig_folded.y_range.start]

        # Map changes
        bls_source.selected.on_change('indices', update_upon_period_selection)
        duration_slider.on_change('value', update_model_slider)
        npoints_slider.on_change('value', update_model_slider)


        # Create the lightcurve figure and its vertical marker
        fig_folded = make_folded_figure_elements(f, f_source, f_model_lc, f_model_lc_source, f_help_source)
        fig_folded.title.text = 'Period: {} days \t T0: {}'.format(np.round(best_period, 7), np.round(best_t0, 5))


        fig_bls, vertical_line, offhand_lines = make_bls_figure_elements(result, bls_source)
        fig_lc = make_lightcurve_figure_elements(lc, model_lc, lc_source, model_lc_source)

        vertical_line.update(location=best_period)
        for idx, line in enumerate(offhand_lines[0:3]):
            line.update(location=best_period/(idx + 2))
        for idx, line in enumerate(offhand_lines[3:]):
            line.update(location=best_period*(idx + 2))



        fig_bls.on_event(PanEnd, update_model_slider_EVENT)
        fig_bls.on_event(Reset, update_model_slider_EVENT)

        fig_folded.on_event(PanEnd, update_folded_plot)
        fig_folded.on_event(Reset, update_folded_plot_reset)


        doc.add_root(layout([[[fig_bls], fig_folded], fig_lc, [Spacer(width=25), duration_slider, Spacer(width=50), npoints_slider]]))

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
