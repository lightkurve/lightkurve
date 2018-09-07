"""Provides tools for interactive visualizations

Example use
-----------

# Must be run from a Jupyter notebook

from lightkurve import KeplerTargetPixelFile
tpf = KeplerTargetPixelFile.from_archive(228682548) # SN 2018 oh for example
tpf.interact()

# An interactive visualization will pop up

"""
from __future__ import division, print_function
import logging
import numpy as np
log = logging.getLogger(__name__)
from .utils import KeplerQualityFlags
try:
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import LogColorMapper, Selection, Slider, RangeSlider, \
                Span, ColorBar, LogTicker, Range1d
    from bokeh.layouts import row, column, widgetbox
    from bokeh.models.tools import HoverTool, LassoSelectTool
    from bokeh.models.formatters import NumeralTickFormatter, PrintfTickFormatter

    output_notebook()
except ImportError:
    log.error("The interact() tool requires `bokeh` to be installed. "
              "bokeh can be installed using `conda install bokeh`.")
    raise


__all__ = []


def map_cadences(tpf, lc):
    """Create a lookup dictionary to map cadences to indices

    Parameters
    ----------
    tpf: TargetPixelFile object

    Returns
    -------
    cadence_lookup, missing_cadences: dict, list
        A dictionary mapping each existing tpf cadence to a TPF slice index;
        A list of cadences for which data is unavailable
    """
    # Map cadence to index for quick array slicing.
    lc_cad_matches = np.in1d(tpf.cadenceno, lc.cadenceno)
    if (lc_cad_matches.sum() != len(lc.cadenceno)) :
        raise ValueError("The lightcurve provided has cadences that are not "
                         "present in the Target Pixel File.")
    min_cadence, max_cadence = np.min(tpf.cadenceno), np.max(tpf.cadenceno)
    cadence_lookup = {cad: j for j, cad in enumerate(tpf.cadenceno)}
    cadence_full_range = np.arange(min_cadence, max_cadence, 1, dtype=np.int)
    missing_cadences = list(set(cadence_full_range)-set(tpf.cadenceno))
    return (cadence_lookup, missing_cadences)


def prepare_lightcurve_datasource(lc):
    """Prepare a bokeh DataSource object for tool tips

    Parameters
    ----------
    lc: LightCurve object

    Returns
    ----------
    bokeh DataSource
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

    source = ColumnDataSource(data=dict(
        time=lc.time,
        time_iso=human_time,
        flux=lc.flux,
        cadence=lc.cadenceno,
        quality_code=lc.quality,
        quality=np.array(qual_strings)))

    return source

def prepare_tpf_datasource(tpf):
    """Prepare a bokeh DataSource object for selection glyphs

    Parameters
    ----------
    tpf: TargetPixelFile

    Returns
    ----------
    bokeh DataSource
    """
    n_pixels = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, n_pixels, 1, dtype=int).reshape(tpf.flux[0, :, :].shape)
    xx = tpf.column + np.arange(tpf.shape[2])
    yy = tpf.row + np.arange(tpf.shape[1])
    xa, ya = np.meshgrid(xx, yy)
    preselection = Selection()
    preselection.indices = pixel_index_array[tpf.pipeline_mask].reshape(-1).tolist()
    source2 = ColumnDataSource(data=dict(xx=xa+0.5, yy=ya+0.5), selected=preselection)
    return source2

def get_lightcurve_y_limits(source):
    """Make the lightcurve figure elements

    Parameters
    ----------
    data_source: `bokeh.models.sources.ColumnDataSource` instance
    """
    sig_lo, med, sig_hi = np.nanpercentile(source.data['flux'], (16, 50, 84))
    robust_sigma = (sig_hi - sig_lo)/2.0
    return med - 5.0 * robust_sigma, med + 5.0 * robust_sigma


def make_lightcurve_figure_elements(lc, source):
    """Make the lightcurve figure elements

    Parameters
    ----------
    tpf: `TargetPixelFile` instance
    source: `bokeh.models.sources.ColumnDataSource` instance

    Returns
    ----------
    fig: `bokeh.plotting.figure` instance
    """
    if lc.mission == 'K2':
        title = "Lightcurve for EPIC {} (K2 C{})".format(
            lc.keplerid, lc.campaign)
    elif lc.mission == 'Kepler':
        title = "Lightcurve for KIC {} (Kepler Q{})".format(
            lc.keplerid, lc.quarter)
    elif lc.mission == 'TESS':
        title = "Lightcurve for TIC {} (TESS Sec. {})".format(
            lc.ticid, lc.sector)
    else:
        title = "Lightcurve for target {}".format(lc.targetid)

    fig = figure(title=title, plot_height=340, plot_width=600,
                 tools="pan,wheel_zoom,box_zoom,reset",
                 toolbar_location="below", logo=None)
    fig.title.offset = -10
    fig.border_fill_color = "whitesmoke"

    fig.yaxis.axis_label = 'Flux (e/s)'
    fig.xaxis.axis_label = 'Time - 2454833 (days)'

    ylims = get_lightcurve_y_limits(source)
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    step_dat = fig.step('time', 'flux', line_width=1, color='gray',
                        source=source, nonselection_line_color='gray')

    circ = fig.circle('time', 'flux', source=source, fill_alpha=0.3, size=8, line_color=None,
                      selection_color="firebrick", nonselection_fill_alpha=0.0,
                      nonselection_fill_color="grey", nonselection_line_color=None,
                      nonselection_line_alpha=0.0, fill_color=None,
                      hover_fill_color="firebrick", hover_alpha=0.9, hover_line_color="white")

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
    vert = Span(location=0, dimension='height', line_color='firebrick',
                line_width=4, line_alpha=0.5)
    fig.add_layout(vert)

    return fig, step_dat, vert



def make_tpf_figure_elements(tpf, source):
    """Make the lightcurve figure elements

    Parameters
    ----------
    tpf: `TargetPixelFile` instance
    source: `bokeh.models.sources.ColumnDataSource` instance

    Returns
    ----------
    fig: `bokeh.plotting.figure` instance
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
                 toolbar_location="below", logo=None)

    fig.yaxis.axis_label = 'Pixel Row Number'
    fig.xaxis.axis_label = 'Pixel Column Number'
    fig.border_fill_color = "whitesmoke"


    pedestal = np.nanmin(tpf.flux)
    vlo, lo, med, hi, vhi = np.nanpercentile(tpf.flux-pedestal, [0.2, 1, 50, 95, 99.8])
    vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    color_mapper = LogColorMapper(palette="Viridis256", low=lo, high=hi)

    fig_dat = fig.image([pedestal + tpf.flux[0, :, :]], x=tpf.column, y=tpf.row,
                        dw=tpf.shape[2], dh=tpf.shape[1], dilate=True,
                        color_mapper=color_mapper)

    # The colorbar will update with the screen stretch slider
    # The colorbar margin increases as the length of the tick labels grows.
    # This colorbar share of the plot window grows, shrinking plot area.
    # This effect is known, some workarounds might work to fix the plot area:
    # https://github.com/bokeh/bokeh/issues/5186
    color_bar = ColorBar(color_mapper=color_mapper,ticker=LogTicker(),
                         label_standoff=-10, border_line_color=None, location=(0,0),
                         background_fill_color='whitesmoke',major_label_text_align = 'left',
                         major_label_text_baseline = 'middle',title = 'e/s', margin = 0)
    fig.add_layout(color_bar, 'left')
    color_bar.formatter = PrintfTickFormatter(format="%14u")#NumeralTickFormatter(format='      0,0')

    pixels = fig.rect('xx', 'yy', 1, 1, source=source, fill_color='gray',
                      fill_alpha=0.4, line_color='white')

    # Lasso Select apparently does not work with boxes/quads/rect.
    # See https://github.com/bokeh/bokeh/issues/964
    #fig.add_tools(LassoSelectTool(renderers=[pixels], select_every_mousemove=False)

    return (fig, fig_dat,
            {'pedestal':pedestal, 'vlo':vlo, 'lo':lo, 'med':med,
             'hi':hi, 'vhi':vhi, 'vstep':vstep})

def pixel_selector_standalone(tpf):
    """Display an interactive IPython Notebook widget to select pixel masks.

    The widget will show both the lightcurve and pixel data.  The pixel data
    supports pixel selection via Bokeh tap and box select tools in an
    interactive javascript user interface.

    Note: at this time, this feature only works inside an active Jupyter
    Notebook, and tends to be too slow when more than ~30,000 cadences
    are contained in the TPF (e.g. short cadence data).

    Parameters
    ----------
    None
    """

    lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)

    n_pixels = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, n_pixels, 1, dtype=int).reshape(tpf.flux[0, :, :].shape)

    # Bokeh cannot handle many data points
    # https://github.com/bokeh/bokeh/issues/7490
    if len(lc.cadenceno) > 30000:
        raise RuntimeError('Interact cannot display more than 30000 cadences.')

    def modify_doc(doc):

        # The data source includes metadata for hover-over tooltips
        source = prepare_lightcurve_datasource(lc)
        source2 = prepare_tpf_datasource(tpf)

        # Lightcurve plot
        fig1, step_dat, vert = make_lightcurve_figure_elements(lc, source)

        # Postage stamp image
        fig2, fig2_dat, stretch = make_tpf_figure_elements(tpf, source2)

        # Interactive slider widgets
        cadence_slider = Slider(start=0, end=len(tpf.time)-1, value=0, step=1.0,
                                title="TPF slice index", width=600)

        screen_slider = RangeSlider(start=np.log10(stretch['vlo']),
                                    end=np.log10(stretch['vhi']),
                                    step=stretch['vstep'],
                                    title="Pixel Stretch (log)",
                                    value=(np.log10(stretch['lo']), np.log10(stretch['hi'])),
                                    width=250)

        existing_selection = source2.selected.to_json(True).copy()

        # Callbacks
        def update_upon_pixel_selection(attr, old, new):
            '''Callback to take action when pixels are selected'''
            #check if a selection was "re-clicked".
            if ((sorted(existing_selection['indices']) == sorted(new.indices)) &
                 (new.indices != [])):
                source2.selected = Selection(indices=new.indices[1:])
                existing_selection['indices'] = new.indices[1:]
            else:
                existing_selection['indices'] = new.indices

            if source2.selected.indices != []:
                selected_indices = np.array(source2.selected.indices)
                selected_mask = np.isin(pixel_index_array, selected_indices)
                lc_new = tpf.to_lightcurve(aperture_mask=selected_mask)
                source.data['flux']= lc_new.flux
                ylims = get_lightcurve_y_limits(source)
                fig1.y_range.start = ylims[0]
                fig1.y_range.end = ylims[1]
            else:
                source.data['flux'] = lc.flux * 0.0
                fig1.y_range.start = -1
                fig1.y_range.end = 1


        def update_upon_cadence_change(attr, old, new):
            '''Callback to take action when cadence slider changes'''
            fig2_dat.data_source.data['image'] = [tpf.flux[new, :, :]
                                                  - stretch['pedestal']]
            vert.update(location=tpf.time[new])

        def update_upon_stetch_change(attr, old, new):
            '''Callback to take action when screen stretch'''
            fig2_dat.glyph.color_mapper.high = 10**new[1]
            fig2_dat.glyph.color_mapper.low = 10**new[0]


        # Map changes to callbacks
        source2.on_change('selected', update_upon_pixel_selection)
        cadence_slider.on_change('value', update_upon_cadence_change)
        screen_slider.on_change('value', update_upon_stetch_change)


        # Layout all of the plots
        row1 = row(fig2, fig1)
        widgets = widgetbox(cadence_slider, screen_slider)
        row_and_col = column(row1, widgets)
        doc.add_root(row_and_col)

    show(modify_doc)


def interact_classic(tpf, lc=None):
    """Display an interactive IPython Notebook widget to inspect the data.

    The widget will show both the lightcurve and pixel data.  By default,
    the lightcurve shown is obtained by calling the `to_lightcurve()` method,
    unless the user supplies a custom `LightCurve` object.

    This feature requires two optional dependencies:
    - bokeh>=0.12.15
    - ipywidgets>=7.2.0
    These can be installed using e.g. `conda install bokeh ipywidgets`.

    Note: at this time, this feature only works inside an active Jupyter
    Notebook, and tends to be too slow when more than ~30,000 cadences
    are contained in the TPF (e.g. short cadence data).

    Parameters
    ----------
    lc : LightCurve object
        An optional pre-processed lightcurve object to show.
    """
    try:
        import ipywidgets as widgets
        from bokeh.io import push_notebook, show, output_notebook
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.models import Span, LogColorMapper
        from bokeh.layouts import row
        from bokeh.models.tools import HoverTool, LassoSelectTool
        from IPython.display import display
        output_notebook()
    except ImportError:
        log.error("The interact() tool requires `bokeh` and `ipywidgets` to be installed. "
                  "These can be installed using `conda install bokeh ipywidgets`.")
        return None

    ytitle = 'Flux'
    if lc is None:
        lc = tpf.to_lightcurve()
        ytitle = 'Flux (e/s)'

    # Bokeh cannot handle many data points
    # https://github.com/bokeh/bokeh/issues/7490
    if len(lc.cadenceno) > 30000:
        raise RuntimeError('Interact cannot display more than 30000 cadences.')

    # Map cadence to index for quick array slicing.
    n_lc_cad = len(lc.cadenceno)
    n_cad, nx, ny = tpf.flux.shape
    lc_cad_matches = np.in1d(tpf.cadenceno, lc.cadenceno)
    if lc_cad_matches.sum() != n_lc_cad:
        raise ValueError("The lightcurve provided has cadences that are not "
                         "present in the Target Pixel File.")
    min_cadence, max_cadence = np.min(tpf.cadenceno), np.max(tpf.cadenceno)
    cadence_lookup = {cad: j for j, cad in enumerate(tpf.cadenceno)}
    cadence_full_range = np.arange(min_cadence, max_cadence, 1, dtype=np.int)
    missing_cadences = list(set(cadence_full_range)-set(tpf.cadenceno))

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

    # Convert time into human readable strings, breaks with NaN time
    # See https://github.com/KeplerGO/lightkurve/issues/116
    if (tpf.time == tpf.time).all():
        human_time = tpf.astropy_time.isot[lc_cad_matches]
    else:
        human_time = [' '] * n_lc_cad

    # Each data source will later become a hover-over tooltip
    source = ColumnDataSource(data=dict(
        time=lc.time,
        time_iso=human_time,
        flux=lc.flux,
        cadence=lc.cadenceno,
        quality_code=lc.quality,
        quality=np.array(qual_strings)))

    # Provide extra metadata in the title
    if tpf.mission == 'K2':
        title = "Quicklook lightcurve for EPIC {} (K2 Campaign {})".format(
            tpf.keplerid, tpf.campaign)
    elif tpf.mission == 'Kepler':
        title = "Quicklook lightcurve for KIC {} (Kepler Quarter {})".format(
            tpf.keplerid, tpf.quarter)
    elif tpf.mission == 'TESS':
        title = "Quicklook lightcurve for TIC {} (TESS Sector {})".format(
            tpf.ticid, tpf.sector)
    else:
        title = "Quicklook lightcurve for target {}".format(tpf.targetid)

    # Figure 1 shows the lightcurve with steps, tooltips, and vertical line
    fig1 = figure(title=title, plot_height=300, plot_width=600,
                  tools="pan,wheel_zoom,box_zoom,save,reset")
    fig1.yaxis.axis_label = ytitle
    fig1.xaxis.axis_label = 'Time [{}]'.format(lc.time_format.upper())
    fig1.step('time', 'flux', line_width=1, color='gray', source=source,
              nonselection_line_color='gray', mode="center")

    r = fig1.circle('time', 'flux', source=source, fill_alpha=0.3, size=8,
                    line_color=None, selection_color="firebrick",
                    nonselection_fill_alpha=0.0, nonselection_line_color=None,
                    nonselection_line_alpha=0.0, fill_color=None,
                    hover_fill_color="firebrick", hover_alpha=0.9,
                    hover_line_color="white")

    fig1.add_tools(HoverTool(tooltips=[("Cadence", "@cadence"),
                                       ("Time ({})".format(lc.time_format.upper()), "@time{0,0.000}"),
                                       ("Time (ISO)", "@time_iso"),
                                       ("Flux", "@flux"),
                                       ("Quality Code", "@quality_code"),
                                       ("Quality Flag", "@quality")],
                             renderers=[r],
                             mode='mouse',
                             point_policy="snap_to_data"))
    # Vertical line to indicate the cadence shown in Fig 2
    vert = Span(location=0, dimension='height', line_color='firebrick',
                line_width=4, line_alpha=0.5)
    fig1.add_layout(vert)

    # Figure 2 shows the Target Pixel File stamp with log screen stretch
    if tpf.mission in ['Kepler', 'K2']:
        title = 'Pixel data (CCD {}.{})'.format(tpf.module, tpf.output)
    elif tpf.mission == 'TESS':
        title = 'Pixel data (Camera {}.{})'.format(tpf.camera, tpf.ccd)
    else:
        title = "Pixel data"
    fig2 = figure(plot_width=300, plot_height=300,
                  tools="pan,wheel_zoom,box_zoom,save,reset",
                  title=title)
    fig2.yaxis.axis_label = 'Pixel Row Number'
    fig2.xaxis.axis_label = 'Pixel Column Number'

    pedestal = np.nanmin(tpf.flux[lc_cad_matches, :, :])
    stretch_dims = np.prod(tpf.flux[lc_cad_matches, :, :].shape)
    screen_stretch = tpf.flux[lc_cad_matches, :, :].reshape(stretch_dims) - pedestal
    screen_stretch = screen_stretch[np.isfinite(screen_stretch)]  # ignore NaNs
    screen_stretch = screen_stretch[screen_stretch > 0.0]
    vlo = np.min(screen_stretch)
    vhi = np.max(screen_stretch)
    vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    lo, med, hi = np.nanpercentile(screen_stretch, [1, 50, 95])
    color_mapper = LogColorMapper(palette="Viridis256", low=lo, high=hi)

    fig2_dat = fig2.image([tpf.flux[0, :, :] - pedestal], x=tpf.column,
                          y=tpf.row, dw=tpf.shape[2], dh=tpf.shape[1],
                          dilate=False, color_mapper=color_mapper)

    # The figures appear before the interactive widget sliders
    show(row(fig1, fig2), notebook_handle=True)

    # The widget sliders call the update function each time
    def update(cadence, log_stretch):
        """Function that connects to the interact widget slider values"""
        fig2_dat.glyph.color_mapper.high = 10**log_stretch[1]
        fig2_dat.glyph.color_mapper.low = 10**log_stretch[0]
        if cadence not in missing_cadences:
            index_val = cadence_lookup[cadence]
            vert.update(line_alpha=0.5)
            if tpf.time[index_val] == tpf.time[index_val]:
                vert.update(location=tpf.time[index_val])
            else:
                vert.update(line_alpha=0.0)
            fig2_dat.data_source.data['image'] = [tpf.flux[index_val, :, :]
                                                  - pedestal]
        else:
            vert.update(line_alpha=0)
            fig2_dat.data_source.data['image'] = [tpf.flux[0, :, :] * np.NaN]
        try:
            push_notebook()
        except AttributeError:
            log.error('ERROR: interact() can only be used inside a Jupyter Notebook.\n')
            return None

    # Define the widgets that enable the interactivity
    play = widgets.Play(interval=10, value=min_cadence, min=min_cadence,
                        max=max_cadence, step=1, description="Press play",
                        disabled=False)
    play.show_repeat, play._repeat = False, False
    cadence_slider = widgets.IntSlider(
        min=min_cadence, max=max_cadence,
        step=1, value=min_cadence, description='Cadence',
        layout=widgets.Layout(width='40%', height='20px'))
    screen_slider = widgets.FloatRangeSlider(
        value=[np.log10(lo), np.log10(hi)],
        min=np.log10(vlo),
        max=np.log10(vhi),
        step=vstep,
        description='Pixel Stretch (log)',
        style={'description_width': 'initial'},
        continuous_update=False,
        layout=widgets.Layout(width='30%', height='20px'))
    widgets.jslink((play, 'value'), (cadence_slider, 'value'))
    ui = widgets.HBox([play, cadence_slider, screen_slider])
    out = widgets.interactive_output(update, {'cadence': cadence_slider,
                                              'log_stretch': screen_slider})
    display(ui, out)
