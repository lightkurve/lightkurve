"""Provides tools for interactive visualizations.

Example use
-----------
The functions in this module are used to create Bokeh-based visualization
widgets.  For example, the following code will create an interactive
visualization widget showing the pixel data and a lightcurve::

    from lightkurve import KeplerTargetPixelFile
    tpf = KeplerTargetPixelFile.from_archive(228682548) # SN 2018 oh for example
    tpf.interact()

Note that this will only work inside a Jupyter notebook at this time.
"""
from __future__ import division, print_function
import logging
import numpy as np
from astropy.stats import sigma_clip
from .utils import KeplerQualityFlags

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
    from bokeh.models.widgets import Button
    from bokeh.models.formatters import PrintfTickFormatter
except ImportError:
    pass  # We will print a nice error message in the `show_interact_widget` function


def prepare_lightcurve_datasource(lc):
    """Prepare a bokeh ColumnDataSource object for tool tips.

    Parameters
    ----------
    lc : LightCurve object
        The light curve to be shown.

    Returns
    -------
    lc_source : bokeh.plotting.ColumnDataSource
    """
    # Convert time into human readable strings, breaks with NaN time
    # See https://github.com/KeplerGO/lightkurve/issues/116
    if (lc.time == lc.time).all():
        human_time = lc.astropy_time.isot
    else:
        human_time = [' '] * len(lc.flux)

    # Convert binary quality numbers into human readable strings
    qual_strings = []
    for bitmask in lc.quality:
        flag_str_list = KeplerQualityFlags.decode(bitmask)
        if len(flag_str_list) == 0:
            qual_strings.append(' ')
        if len(flag_str_list) == 1:
            qual_strings.append(flag_str_list[0])
        if len(flag_str_list) > 1:
            qual_strings.append("; ".join(flag_str_list))

    lc_source = ColumnDataSource(data=dict(
                                 time=lc.time,
                                 time_iso=human_time,
                                 flux=lc.flux,
                                 cadence=lc.cadenceno,
                                 quality_code=lc.quality,
                                 quality=np.array(qual_strings)))
    return lc_source


def prepare_tpf_datasource(tpf):
    """Prepare a bokeh DataSource object for selection glyphs

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to be shown.

    Returns
    -------
    tpf_source : bokeh.plotting.ColumnDataSource
        Bokeh object to be shown.
    """
    n_pixels = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, n_pixels, 1, dtype=int).reshape(tpf.flux[0, :, :].shape)
    xx = tpf.column + np.arange(tpf.shape[2])
    yy = tpf.row + np.arange(tpf.shape[1])
    xa, ya = np.meshgrid(xx, yy)
    preselection = Selection()
    preselection.indices = pixel_index_array[tpf.pipeline_mask].reshape(-1).tolist()
    tpf_source = ColumnDataSource(data=dict(xx=xa+0.5, yy=ya+0.5), selected=preselection)
    return tpf_source


def get_lightcurve_y_limits(lc_source):
    """Compute sensible defaults for the Y axis limits of the lightcurve plot.

    Parameters
    ----------
    lc_source : bokeh.plotting.ColumnDataSource
        The lightcurve being shown.

    Returns
    -------
    ymin, ymax : float, float
        Flux min and max limits.
    """
    flux = sigma_clip(lc_source.data['flux'], sigma=5)
    low, high = np.nanpercentile(flux, (1, 99))
    margin = 0.10 * (high - low)
    return low - margin, high + margin


def make_lightcurve_figure_elements(lc, lc_source):
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

    fig = figure(title=title, plot_height=340, plot_width=600,
                 tools="pan,wheel_zoom,box_zoom,tap,reset",
                 toolbar_location="below", logo=None,
                 border_fill_color="whitesmoke")
    fig.title.offset = -10
    fig.yaxis.axis_label = 'Flux (e/s)'
    fig.xaxis.axis_label = 'Time - 2454833 (days)'

    ylims = get_lightcurve_y_limits(lc_source)
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add step lines, circles, and hover-over tooltips
    fig.step('time', 'flux', line_width=1, color='gray',
             source=lc_source, nonselection_line_color='gray',
             nonselection_line_alpha=1.0)
    circ = fig.circle('time', 'flux', source=lc_source, fill_alpha=0.3, size=8,
                      line_color=None, selection_color="firebrick",
                      nonselection_fill_alpha=0.0, nonselection_fill_color="grey",
                      nonselection_line_color=None, nonselection_line_alpha=0.0,
                      fill_color=None, hover_fill_color="firebrick",
                      hover_alpha=0.9, hover_line_color="white")
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

    # Vertical line to indicate the cadence
    vertical_line = Span(location=lc.time[0], dimension='height',
                         line_color='firebrick', line_width=4, line_alpha=0.5)
    fig.add_layout(vertical_line)

    return fig, vertical_line


def make_tpf_figure_elements(tpf, tpf_source, pedestal=0):
    """Returns the lightcurve figure elements.

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to show.
    tpf_source : bokeh.plotting.ColumnDataSource
        TPF data source.

    Returns
    -------
    fig, stretch_slider : bokeh.plotting.figure.Figure, RangeSlider
    """
    if tpf.mission in ['Kepler', 'K2']:
        title = 'Pixel data (CCD {}.{})'.format(tpf.module, tpf.output)
    elif tpf.mission == 'TESS':
        title = 'Pixel data (Camera {}.{})'.format(tpf.camera, tpf.ccd)
    else:
        title = "Pixel data"

    fig = figure(plot_width=370, plot_height=340,
                 x_range=(tpf.column, tpf.column+tpf.shape[2]),
                 y_range=(tpf.row, tpf.row+tpf.shape[1]),
                 title=title, tools='tap,box_select,wheel_zoom,reset',
                 toolbar_location="below", logo=None,
                 border_fill_color="whitesmoke")

    fig.yaxis.axis_label = 'Pixel Row Number'
    fig.xaxis.axis_label = 'Pixel Column Number'

    vlo, lo, hi, vhi = np.nanpercentile(tpf.flux - pedestal, [0.2, 1, 95, 99.8])
    vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    color_mapper = LogColorMapper(palette="Viridis256", low=lo, high=hi)

    fig.image([pedestal + tpf.flux[0, :, :]], x=tpf.column, y=tpf.row,
              dw=tpf.shape[2], dh=tpf.shape[1], dilate=True,
              color_mapper=color_mapper, name="tpfimg")

    # The colorbar will update with the screen stretch slider
    # The colorbar margin increases as the length of the tick labels grows.
    # This colorbar share of the plot window grows, shrinking plot area.
    # This effect is known, some workarounds might work to fix the plot area:
    # https://github.com/bokeh/bokeh/issues/5186
    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(desired_num_ticks=8),
                         label_standoff=-10, border_line_color=None, location=(0, 0),
                         background_fill_color='whitesmoke', major_label_text_align='left',
                         major_label_text_baseline='middle', title='e/s', margin=0)
    fig.add_layout(color_bar, 'right')

    color_bar.formatter = PrintfTickFormatter(format="%14u")

    fig.rect('xx', 'yy', 1, 1, source=tpf_source, fill_color='gray',
             fill_alpha=0.4, line_color='white')

    # Configure the stretch slider and its callback function
    stretch_slider = RangeSlider(start=np.log10(vlo),
                                 end=np.log10(vhi),
                                 step=vstep,
                                 title='Screen Stretch (log)',
                                 value=(np.log10(lo), np.log10(hi)),
                                 orientation='horizontal',
                                 width=200,
                                 direction='ltr',
                                 show_value=True,
                                 sizing_mode='fixed',
                                 name='tpfstretch')

    def stretch_change_callback(attr, old, new):
        """TPF stretch slider callback."""
        fig.select('tpfimg')[0].glyph.color_mapper.high = 10**new[1]
        fig.select('tpfimg')[0].glyph.color_mapper.low = 10**new[0]

    stretch_slider.on_change('value', stretch_change_callback)

    return fig, stretch_slider


def show_interact_widget(tpf, lc=None, notebook_url='localhost:8888', max_cadences=30000):
    """Display an interactive Jupyter Notebook widget to inspect the pixel data.

    The widget will show both the lightcurve and pixel data.  The pixel data
    supports pixel selection via Bokeh tap and box select tools in an
    interactive javascript user interface.

    Note: at this time, this feature only works inside an active Jupyter
    Notebook, and tends to be too slow when more than ~30,000 cadences
    are contained in the TPF (e.g. short cadence data).

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Target Pixel File to interact with
    notebook_url: str
        Location of the Jupyter notebook page (default: "localhost:8888")
        When showing Bokeh applications, the Bokeh server must be
        explicitly configured to allow connections originating from
        different URLs. This parameter defaults to the standard notebook
        host and port. If you are running on a different location, you
        will need to supply this value for the application to display
        properly. If no protocol is supplied in the URL, e.g. if it is
        of the form "localhost:8888", then "http" will be used.
    max_cadences : int
        Raise a RuntimeError if the number of cadences shown is larger than
        this value. This limit helps keep browsers from becoming unresponsive.
    """
    try:
        import bokeh
    except ImportError:
        log.error("The interact() tool requires the `bokeh` package; "
                  "you can install bokeh using e.g. `conda install bokeh`.")
        return None

    if lc is None:
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    else:
        if len(lc.time) != len(tpf.time):
            log.error("The custom lightcurve provided to interact() does not contain "
                      "the same number of cadences ({}) as the target pixel file ({})."
                      "".format(len(lc.time), len(tpf.time)))
            return None

    n_pixels = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, n_pixels, 1, dtype=int).reshape(tpf.flux[0, :, :].shape)

    # Bokeh cannot handle many data points
    # https://github.com/bokeh/bokeh/issues/7490
    if len(lc.cadenceno) > max_cadences:
        raise RuntimeError('Interact cannot display more than {} cadences.'.format(max_cadences))

    def create_interact_ui(doc):
        # The data source includes metadata for hover-over tooltips
        lc_source = prepare_lightcurve_datasource(lc)
        tpf_source = prepare_tpf_datasource(tpf)

        # Create the lightcurve figure and its vertical marker
        fig_lc, vertical_line = make_lightcurve_figure_elements(lc, lc_source)

        # Create the TPF figure and its stretch slider
        pedestal = np.nanmin(tpf.flux)
        fig_tpf, stretch_slider = make_tpf_figure_elements(tpf, tpf_source, pedestal=pedestal)

        # Helper lookup table which maps cadence number onto flux array index.
        tpf_index_lookup = {cad: idx for idx, cad in enumerate(tpf.cadenceno)}

        # Interactive slider widgets and buttons to select the cadence number
        cadence_slider = Slider(start=np.min(tpf.cadenceno),
                                end=np.max(tpf.cadenceno),
                                value=np.min(tpf.cadenceno),
                                step=1,
                                title="Cadence Number",
                                width=490)
        r_button = Button(label=">", button_type="default", width=30)
        l_button = Button(label="<", button_type="default", width=30)

        existing_selection = tpf_source.selected.to_json(True).copy()

        # Callbacks
        def update_upon_pixel_selection(attr, old, new):
            """Callback to take action when pixels are selected."""
            # check if a selection was "re-clicked".
            if ((sorted(existing_selection['indices']) == sorted(new.indices)) &
                    (new.indices != [])):
                tpf_source.selected = Selection(indices=new.indices[1:])
                existing_selection['indices'] = new.indices[1:]
            else:
                existing_selection['indices'] = new.indices

            if tpf_source.selected.indices != []:
                selected_indices = np.array(tpf_source.selected.indices)
                selected_mask = np.isin(pixel_index_array, selected_indices)
                lc_new = tpf.to_lightcurve(aperture_mask=selected_mask)
                lc_source.data['flux'] = lc_new.flux
                ylims = get_lightcurve_y_limits(lc_source)
                fig_lc.y_range.start = ylims[0]
                fig_lc.y_range.end = ylims[1]
            else:
                lc_source.data['flux'] = lc.flux * 0.0
                fig_lc.y_range.start = -1
                fig_lc.y_range.end = 1

        def update_upon_cadence_change(attr, old, new):
            '''Callback to take action when cadence slider changes'''
            if new in tpf.cadenceno:
                frameno = tpf_index_lookup[new]
                fig_tpf.select('tpfimg')[0].data_source.data['image'] = [tpf.flux[frameno, :, :]
                                                                         - pedestal]
                vertical_line.update(location=tpf.time[frameno])
            else:
                fig_tpf.select('tpfimg')[0].data_source.data['image'] = [tpf.flux[0, :, :] * np.NaN]
            lc_source.selected.indices = []

        def go_right_by_one():
            existing_value = cadence_slider.value
            if existing_value < np.max(tpf.cadenceno):
                cadence_slider.value = existing_value + 1

        def go_left_by_one():
            existing_value = cadence_slider.value
            if existing_value > np.min(tpf.cadenceno):
                cadence_slider.value = existing_value - 1

        def jump_to_lightcurve_position(attr, old, new):
            if new.indices != []:
                cadence_slider.value = lc.cadenceno[new.indices[0]]

        # Map changes to callbacks
        r_button.on_click(go_right_by_one)
        l_button.on_click(go_left_by_one)
        tpf_source.on_change('selected', update_upon_pixel_selection)
        lc_source.on_change('selected', jump_to_lightcurve_position)
        cadence_slider.on_change('value', update_upon_cadence_change)

        # Layout all of the plots
        space1, space2, space3 = Spacer(width=15), Spacer(width=30), Spacer(width=80)
        widgets_and_figures = layout([fig_lc, fig_tpf],
                                     [l_button, space1, r_button, space2,
                                      cadence_slider, space3, stretch_slider])
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
