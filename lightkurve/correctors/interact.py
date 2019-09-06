"""Provides tools for interactive visualizations for correctors
"""
from __future__ import division, print_function
import logging
import warnings
import numpy as np
from astropy.stats import sigma_clip
from ..utils import KeplerQualityFlags, LightkurveWarning
from ..interact import prepare_lightcurve_datasource, get_lightcurve_y_limits
import os


from . import SFFCorrector

log = logging.getLogger(__name__)

# Import the optional Bokeh dependency, or print a friendly error otherwise.
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import LogColorMapper, Selection, Slider, RangeSlider, \
        Span, ColorBar, LogTicker, Range1d
    from bokeh.layouts import layout, Spacer
    from bokeh.models.tools import HoverTool
    from bokeh.models.widgets import Button, RadioButtonGroup, CheckboxGroup
    from bokeh.models.formatters import PrintfTickFormatter
except ImportError:
    # We will print a nice error message in the `show_interact_widget` function
    pass



def make_lightcurve_figure_elements(lc, lc_source, tools="pan,wheel_zoom,box_zoom,tap,reset",
                                    tooltips=True, line=True, figsize=(340, 600)):
    """Make the lightcurve figure elements.

    Parameters
    ----------
    lc : LightCurve
        Lightcurve to be shown.
    lc_source : bokeh.plotting.ColumnDataSource
        Bokeh object that enables the visualization.

    Returns
    ----------
    fig : `bokeh.plotting.figure` instance
    step_renderer : GlyphRenderer
    vertical_line : Span
    """
    if lc.mission == 'K2':
        title = "Lightcurve for {} (K2 C{})".format(
            lc.label, lc.campaign)
    elif lc.mission == 'Kepler':
        title = "Lightcurve for {} (Kepler Q{})".format(
            lc.label, lc.quarter)
    elif lc.mission == 'TESS':
        title = "Lightcurve for {} (TESS Sec. {})".format(
            lc.label, lc.sector)
    else:
        title = "Lightcurve for target {}".format(lc.label)

    fig = figure(title=title, plot_height=figsize[0], plot_width=figsize[1],
                 tools=tools,
                 toolbar_location="below",
                 border_fill_color="whitesmoke")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux (e/s)'
    fig.xaxis.axis_label = 'Time (days)'
    try:
        if (lc.mission == 'K2') or (lc.mission == 'Kepler'):
            fig.xaxis.axis_label = 'Time - 2454833 (days)'
        elif lc.mission == 'TESS':
            fig.xaxis.axis_label = 'Time - 2457000 (days)'
    except AttributeError:  # no mission keyword available
        pass

    ylims = get_lightcurve_y_limits(lc_source)
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add step lines, circles, and hover-over tooltips
    fig.step('time', 'flux', line_width=1, color='gray',
             source=lc_source, nonselection_line_color='gray',
             nonselection_line_alpha=1.0)
    circ = fig.circle('time', 'flux', source=lc_source, fill_alpha=0.3, size=8,
                      line_color=None, selection_color="firebrick",
                      nonselection_fill_alpha=0.0,
                      nonselection_fill_color="grey",
                      nonselection_line_color=None,
                      nonselection_line_alpha=0.0,
                      fill_color=None, hover_fill_color="firebrick",
                      hover_alpha=0.9, hover_line_color="white")
    if tooltips:
        # YEAHHHHHH
        fig.add_tools(HoverTool(tooltips=[("Cadence", "@cadence"),
                                          ("Time ({})".format(lc.time_format.upper()),
                                           "@time{0,0.000}"),
                                          ("Time (ISO)", "@time_iso"),
                                          ("Flux", "@flux"),
                                          ("Quality Code", "@quality_code"),
                                          ("Quality Flag", "@quality")],
                                renderers=[circ],
                                mode='mouse',
                                point_policy="snap_to_data"))

    if line:
        # Vertical line to indicate the cadence
        vertical_line = Span(location=lc.time[0], dimension='height',
                             line_color='firebrick', line_width=4, line_alpha=0.5)
        fig.add_layout(vertical_line)
        return fig, vertical_line
    return fig

def make_default_export_name(tpf, suffix='custom-lc'):
    """makes the default name to save a custom intetract mask"""
    fn = tpf.hdu.filename()
    if fn is None:
        outname = "{}_{}_{}.fits".format(tpf.mission, tpf.targetid, suffix)
    else:
        base = os.path.basename(fn)
        outname = base.rsplit('.fits')[0] + '-{}.fits'.format(suffix)
    return outname

def show_SFF_interact_widget(corr, notebook_url='localhost:8888', postprocessing=None):
    '''Show an interactive SFF widget...

    Parameters
    ----------

    Returns
    -------


    '''

    lc_source = prepare_lightcurve_datasource(corr.lc)
    def create_interact_ui(doc):
        SFF = corr

        # WIDGETS
        #-----------------------------
        # Window Slider
        window_slider = Slider(start=1,
                                end=50,
                                value=0,
                                step=1,
                                title="Window Number",
                                width=600)

        # Bin Slider
        bin_slider = Slider(start=1,
                                end=50,
                                value=0,
                                step=1,
                                title="Bin Number",
                                width=600)


        # Knot slider
        timescale_slider = Slider(start=0.1,
                                end=10,
                                value=1.5,
                                step=0.1,
                                title="Spline Timescale",
                                width=600)


        niters_button = RadioButtonGroup(labels=["1 Iter", "2 Iter", "3 Iter", "4 Iter", "5 Iter", "6 Iter", "7 Iter", "8 Iter"], active=0)
        preserve_trend = CheckboxGroup(labels=["Preserve Trend"], active=[])
        show_windows = CheckboxGroup(labels=["Show Window Edges"], active=[])
        #-----------------------------

        # Make plot
        fig_lc = make_lightcurve_figure_elements(corr.lc, lc_source, line=False, tools="pan,wheel_zoom,box_zoom,reset", tooltips=False)

        #Make INVISIBLE lines.
        line_dict = {}
        for i in range(50):
            line_dict[i] = Span(location=0, dimension='height',
                                 line_color='firebrick', line_width=0, line_alpha=0.6)
            fig_lc.add_layout(line_dict[i])

        # Callback
        def compute(attr, old, new):
            '''When sliders change, compute the new correction'''
            if (window_slider.value != 0) & (bin_slider.value!=0):
                lc = corr.correct(windows=window_slider.value, bins=bin_slider.value,
                                    niters=niters_button.active + 1,
                                    preserve_trend = bool(len(preserve_trend.active)),
                                    timescale=timescale_slider.value)

                if postprocessing is not None:
                    if not callable(postprocessing):
                        raise ValueError('Post Processing must be a function.')
                    lc = postprocessing(lc)
                    show_windows.active=[]
                    preserve_trend.active=[]
                    preserve_trend.disabled=True
                    show_windows.disabled=True

                new_source = prepare_lightcurve_datasource(lc)
                for k in new_source.data.keys():
                    lc_source.data[k] = new_source.data[k]
                lc_source.data['quality'] = lc.quality
            else:
                lc_source.data['flux'] = corr.lc.flux


        def do_lines(attr, old, new):
            '''When window or window shift changes, change the lines
            '''
            windows = window_slider.value
            idx = 0
            if bool(len(show_windows.active)):
                time = corr.window_points
                for idx, t in enumerate(time):
                    line_dict[idx].update(location=corr.lc.time[t])
                    line_dict[idx].update(line_width=1.5)
            for jdx in range(idx, 50):
                line_dict[jdx].update(location=0)
                line_dict[idx].update(line_width=0)


        window_slider.on_change('value', compute)
        bin_slider.on_change('value', compute)
        timescale_slider.on_change('value', compute)
        niters_button.on_change('active', compute)
        preserve_trend.on_change('active', compute)

        show_windows.on_change('active', do_lines)
        window_slider.on_change('value', do_lines)

        doc.add_root(layout([fig_lc, Spacer(width=30), [niters_button, preserve_trend, show_windows]], [window_slider], [bin_slider], [timescale_slider]))

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
