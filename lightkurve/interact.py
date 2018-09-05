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
                Span, ColorBar, LogTicker
    from bokeh.layouts import row, column, widgetbox
    from bokeh.models.widgets import CheckboxGroup, Toggle
    from bokeh.models.tools import HoverTool
    output_notebook()
except ImportError:
    log.error("The interact() tool requires `bokeh` to be installed. "
              "These can be installed using `conda install bokeh`.")
    raise


__all__ = []



def pixel_selector_standalone(tpf):
    """Display an interactive IPython Notebook widget to select pixel masks.

    The widget will show both the lightcurve and pixel data.  The pixel data
    supports pixel selection via Bokeh tap and box select tools in an
    interactive javascript user interface similar to `.interact()`.

    This feature requires one optional dependency:
    - bokeh>=0.12.15
    Which can be installed using e.g. `conda install bokeh`.

    Note: at this time, this feature only works inside an active Jupyter
    Notebook, and tends to be too slow when more than ~30,000 cadences
    are contained in the TPF (e.g. short cadence data).

    Parameters
    ----------
    None
    """

    lc = tpf.to_lightcurve()
    ytitle = 'Flux (e/s)'
    xx=tpf.column + np.arange(tpf.shape[2])
    yy=tpf.row + np.arange(tpf.shape[1])
    xa, ya = np.meshgrid(xx, yy)

    n_pixels = tpf.flux[0,:,:].size
    pixel_index_array = np.arange(0,n_pixels, 1, dtype=int).reshape(tpf.flux[0,:,:].shape)

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


    def modify_doc(doc):
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

        ymax = np.nanpercentile(tpf.to_lightcurve(aperture_mask='all').flux, 80)*1.2
        fig1 = figure(title=title, plot_height=300, plot_width=550, y_range=(0, ymax),
                   tools="pan,wheel_zoom,box_zoom,reset")#, theme=theme)
        fig1.yaxis.axis_label = 'Normalized Flux'
        fig1.xaxis.axis_label = 'Time - 2454833 (days)'
        step_dat = fig1.step('time', 'flux', line_width=1, color='gray', source=source, nonselection_line_color='gray')

        r = fig1.circle('time', 'flux', source=source, fill_alpha=0.3, size=8,line_color=None,
                     selection_color="firebrick", nonselection_fill_alpha=0.0,
                     nonselection_fill_color="grey",nonselection_line_color=None,
                     nonselection_line_alpha=0.0, fill_color=None,
                     hover_fill_color="firebrick",hover_alpha=0.9,hover_line_color="white")

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


        fig2 = figure(plot_width=370, plot_height=300, x_range=(min(xx), max(xx+1)),
                    y_range=(min(yy), max(yy+1)), title='Target Pixel File',
                    tools='tap, box_select, wheel_zoom, reset')
        fig2.yaxis.axis_label = 'Pixel Row Number'
        fig2.xaxis.axis_label = 'Pixel Column Number'

        pedestal = np.nanmin(tpf.flux)
        vlo, lo, med, hi, vhi = np.nanpercentile(tpf.flux-pedestal, [0.2, 1, 50, 95, 99.8])
        vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
        color_mapper = LogColorMapper(palette="Viridis256", low=lo, high=hi)

        fig2_dat = fig2.image([pedestal+tpf.flux[0,:,:]], x=tpf.column, y=tpf.row,
                          dw=tpf.shape[2], dh=tpf.shape[1], dilate=True,
                          color_mapper=color_mapper)

        preselection = Selection()
        preselection.indices=pixel_index_array[tpf.pipeline_mask].reshape(-1).tolist()
        source2 = ColumnDataSource(data=dict(xx=xa+0.5, yy=ya+0.5), selected=preselection)
        r1 = fig2.rect('xx', 'yy', 1, 1, source=source2, fill_color='gray', fill_alpha=0.4, line_color='white')

        def callback(attr, old, new):
            if len(source2.selected.indices) > 0:
                selected_indices = np.array(source2.selected.indices)
                selected_mask = np.isin(pixel_index_array, selected_indices)
                lc_new = tpf.to_lightcurve(aperture_mask=selected_mask)
                source.data = dict(time=lc.time, flux=lc_new.flux,
                                cadence=lc.cadenceno, quality=lc.quality)
            else:
                source.data = dict(time=lc.time, flux=lc.flux*0.0,
                                cadence=lc.cadenceno, quality=lc.quality)
            callback4('junk')

        source2.on_change('selected', callback)

        n_cadences = len(tpf.time)
        amp_slider = Slider(start=0, end=n_cadences-1, value=0, step=1,
                title="TPF slice index", width = 600)

        def callback2(attr, old, new):
            fig2_dat.data_source.data['image'] = [tpf.flux[new, :, :]
                                                      - pedestal]
            vert.update(location=tpf.time[new])

        amp_slider.on_change('value', callback2)

        def callback3(attr, old, new):
            fig2_dat.glyph.color_mapper.high = 10**new[1]
            fig2_dat.glyph.color_mapper.low = 10**new[0]

        screen_slider = RangeSlider(start=np.log10(vlo), end=np.log10(vhi),
            step=vstep, title="Pixel Stretch (log)",
            value=(np.log10(lo), np.log10(hi)), width = 250)

        screen_slider.on_change('value', callback3)

        checkbox_group = CheckboxGroup(
        labels=["Normalize Lightcurve"])#, "Overplot Complement"])

        def callback4(new):
            if 0 in checkbox_group.active:
                source.data['flux'] = source.data['flux'] / np.nanmedian(source.data['flux'])
                sig_lo, med, sig_hi = np.nanpercentile(source.data['flux'], (16,50,84))
                robust_sigma = (sig_hi - sig_lo)/2.0
                fig1.y_range.start = med-5.0*robust_sigma
                fig1.y_range.end   = med+5.0*robust_sigma
            else:
                fig1.y_range.start = 0.0
                fig1.y_range.end   = ymax
            #if 1 in checkbox_group.active:
            #    selected_indices = np.array(source2.selected.indices)
            #    nonselected_mask = ~np.isin(pixel_index_array, selected_indices)
            #    lc_bak = tpf.to_lightcurve(aperture_mask=nonselected_mask)
            #    step_dat2 = fig1.step(lc_bak.time, lc_bak.flux, line_width=1, color='red')

        checkbox_group.on_click(callback4)

        toggle = Toggle(label="Save Mask", button_type="success")

        def callback5(new):
            mask_out = "mask_ID_{}_C{}.npy".format(tpf.targetid, tpf.campaign)
            selected_indices = np.array(source2.selected.indices)
            selected_mask = np.isin(pixel_index_array, selected_indices)
            np.save(mask_out, selected_mask)

        toggle.on_click(callback5)

        color_bar = ColorBar(color_mapper=color_mapper,ticker=LogTicker(), label_standoff=12, border_line_color=None, location=(0,0))
        fig2.add_layout(color_bar, 'right')


        row1 = row(fig1, fig2)
        widgets = widgetbox(amp_slider, screen_slider, checkbox_group, toggle)
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
        from bokeh.models.tools import HoverTool
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
