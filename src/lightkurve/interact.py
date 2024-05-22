"""Provides tools for interactive visualizations.

Example use
-----------
The functions in this module are used to create Bokeh-based visualization
widgets.  For example, the following code will create an interactive
visualization widget showing the pixel data and a lightcurve::

    # SN 2018 oh Supernova example
    from lightkurve import KeplerTargetPixelFile
    tpf = KeplerTargetPixelFile.from_archive(228682548)
    tpf.interact()

Note that this will only work inside a Jupyter notebook at this time.
"""
from __future__ import division, print_function
import os
import logging
import warnings

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.io import ascii
from astropy.stats import sigma_clip
from astropy.table import Table, Column, MaskedColumn
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning

from .utils import KeplerQualityFlags, LightkurveWarning, LightkurveError, finalize_notebook_url

log = logging.getLogger(__name__)

# Import the optional Bokeh dependency, or print a friendly error otherwise.
_BOKEH_IMPORT_ERROR = None
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import (
        LogColorMapper,
        Slider,
        RangeSlider,
        Span,
        ColorBar,
        LogTicker,
        Range1d,
        LinearColorMapper,
        BasicTicker,
        Arrow,
        VeeHead,
        InlineStyleSheet,
    )
    from bokeh.layouts import layout, Spacer
    from bokeh.models.tools import HoverTool
    from bokeh.models.widgets import Button, Div, CheckboxGroup
    from bokeh.models.formatters import PrintfTickFormatter
except Exception as e:
    # We will print a nice error message in the `show_interact_widget` function
    # the error would be raised there in case users need to diagnose problems
    _BOKEH_IMPORT_ERROR = e


def _fill_masked_or_nan_with(ary, fill_value):
    if not isinstance(ary, np.ndarray):
        # assume to be scalar
        if np.isnan(ary) or ary is np.ma.masked:
            return fill_value
        else:
            return ary

    # case ndarray (also astropy Column)
    ary = ary.copy()
    ary[np.isnan(ary)] = fill_value
    if isinstance(ary, np.ma.MaskedArray):  # also astropy MaskedColumn
        ary = ary.filled(fill_value)
    return ary


def _correct_with_proper_motion(ra, dec, pm_ra, pm_dec, equinox, new_time):
    """Return proper-motion corrected RA / Dec.
       It also return whether proper motion correction is applied or not."""
    # all parameters have units

    if ra is None or dec is None or \
       pm_ra is None or pm_dec is None or (np.all(pm_ra == 0) and np.all(pm_dec == 0)) or \
       equinox is None:
        return ra, dec, False

    # handle cases pm_ra, pm_dec has some nan or masked value
    # we treat them as 0 for the PM correction below
    # (otherwise the result would be nan)
    pm_ra = _fill_masked_or_nan_with(pm_ra, 0.0)
    pm_dec = _fill_masked_or_nan_with(pm_dec, 0.0)

    # To be more accurate, we should have supplied distance to SkyCoord
    # in theory, for Gaia DR2 data, we can infer the distance from the parallax provided.
    # It is not done for 2 reasons:
    # 1. Gaia DR2 data has negative parallax values occasionally. Correctly handling them could be tricky. See:
    #    https://www.cosmos.esa.int/documents/29201/1773953/Gaia+DR2+primer+version+1.3.pdf/a4459741-6732-7a98-1406-a1bea243df79
    # 2. For our purpose (plotting in various interact usage) here, the added distance does not making
    #    noticeable significant difference. E.g., applying it to Proxima Cen, a target with large parallax
    #    and huge proper motion, does not change the result in any noticeable way.
    #
    c = SkyCoord(ra, dec, pm_ra_cosdec=pm_ra, pm_dec=pm_dec,
                frame='icrs', obstime=equinox)

    with warnings.catch_warnings():
        # Suppress ErfaWarning temporarily as a workaround for:
        #   https://github.com/astropy/astropy/issues/11747
        # the same warning appears both as an ErfaWarning and a astropy warning
        # so we filter by the message instead
        warnings.filterwarnings("ignore", message="ERFA function")

        # ignore RuntimeWarning: invalid value encountered in pmsafe
        # as some rows could have missing PM
        warnings.filterwarnings("ignore", message="invalid value encountered in pmsafe")
        new_c = c.apply_space_motion(new_obstime=new_time)
    return new_c.ra, new_c.dec, True


def _get_corrected_coordinate(tpf_or_lc):
    """Extract coordinate from Kepler/TESS FITS, with proper motion corrected
       to the start of observation if proper motion is available."""
    h = tpf_or_lc.meta
    new_time = tpf_or_lc.time[0]

    ra = h.get("RA_OBJ")
    dec = h.get("DEC_OBJ")

    pm_ra = h.get("PMRA")
    pm_dec = h.get("PMDEC")
    equinox = h.get("EQUINOX")

    if ra is None or dec is None or pm_ra is None or pm_dec is None or equinox is None:
        # case cannot apply proper motion due to missing parameters
        return ra, dec, False

    # Note: it'd be better / extensible if the unit is a property of the tpf or lc
    if tpf_or_lc.meta.get("TICID") is not None:
        pm_unit = u.milliarcsecond / u.year
    else:  # assumes to be Kepler / K2
        pm_unit = u.arcsecond / u.year

    ra_corrected, dec_corrected, pm_corrected = _correct_with_proper_motion(
            ra * u.deg, dec *u.deg,
            pm_ra * pm_unit, pm_dec * pm_unit,
            # we assume the data is in J2000 epoch
            Time('2000', format='byear'),
            new_time)
    return ra_corrected.to(u.deg).value,  dec_corrected.to(u.deg).value, pm_corrected


def _to_unitless(items):
    """Convert the values in the item list to unitless one"""
    return [getattr(item, "value", item) for item in items]


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
    # See https://github.com/lightkurve/lightkurve/issues/116
    if (lc.time == lc.time).all():
        human_time = lc.time.isot
    else:
        human_time = [" "] * len(lc.flux)

    # Convert binary quality numbers into human readable strings
    qual_strings = []
    for bitmask in lc.quality:
        if isinstance(bitmask, u.Quantity):
            bitmask = bitmask.value
        flag_str_list = KeplerQualityFlags.decode(bitmask)
        if len(flag_str_list) == 0:
            qual_strings.append(" ")
        if len(flag_str_list) == 1:
            qual_strings.append(flag_str_list[0])
        if len(flag_str_list) > 1:
            qual_strings.append("; ".join(flag_str_list))

    lc_source = ColumnDataSource(
        data=dict(
            time=lc.time.value,
            time_iso=human_time,
            flux=lc.flux.value,
            cadence=lc.cadenceno.value,
            quality_code=lc.quality.value,
            quality=np.array(qual_strings),
        )
    )
    return lc_source


def aperture_mask_to_selected_indices(aperture_mask):
    """Convert the 2D aperture mask to 1D selection indices, for the use with bokeh ColumnDataSource."""
    npix = aperture_mask.size
    pixel_index_array = np.arange(0, npix, 1)
    return pixel_index_array[aperture_mask.reshape(-1)]


def aperture_mask_from_selected_indices(selected_pixel_indices, tpf):
    """Convert an aperture mask in 1D selection indices back to 2D (in the shape of the given TPF)."""
    npix = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, npix, 1).reshape(tpf.flux[0].shape)
    selected_indices = np.array(selected_pixel_indices)
    selected_mask_1d = np.isin(pixel_index_array, selected_indices)
    return selected_mask_1d.reshape(tpf.flux[0].shape)


def prepare_tpf_datasource(tpf, aperture_mask):
    """Prepare a bokeh DataSource object for selection glyphs

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to be shown.
    aperture_mask : boolean numpy array
        The Aperture mask applied at the startup of interact

    Returns
    -------
    tpf_source : bokeh.plotting.ColumnDataSource
        Bokeh object to be shown.
    """
    _, ny, nx = tpf.shape
    # (xa, ya) pair enumerates all pixels of the tpf
    xx = tpf.column + np.arange(nx)
    yy = tpf.row + np.arange(ny)
    xa, ya = np.meshgrid(xx, yy)
    # flatten them, as column data source requires 1d data
    xa = xa.flatten()
    ya = ya.flatten()
    tpf_source = ColumnDataSource(data=dict(xx=xa.astype(float), yy=ya.astype(float)))
    # convert the ndarray from aperture_mask_to_selected_indices() to plain list
    # because bokeh v3.0.2 does not accept ndarray (and causes js error)
    # see https://github.com/bokeh/bokeh/issues/12624
    tpf_source.selected.indices = list(aperture_mask_to_selected_indices(aperture_mask))
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
    with warnings.catch_warnings():  # Ignore warnings due to NaNs
        warnings.simplefilter("ignore", AstropyUserWarning)
        flux = sigma_clip(lc_source.data["flux"], sigma=5, masked=False)
    low, high = np.nanpercentile(flux, (1, 99))
    margin = 0.10 * (high - low)
    return low - margin, high + margin


def make_lightcurve_figure_elements(lc, lc_source, ylim_func=None):
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
    mission = lc.meta.get("MISSION")
    if mission == "K2":
        title = "Lightcurve for {} (K2 C{})".format(lc.label, lc.campaign)
    elif mission == "Kepler":
        title = "Lightcurve for {} (Kepler Q{})".format(lc.label, lc.quarter)
    elif mission == "TESS":
        title = "Lightcurve for {} (TESS Sec. {})".format(lc.label, lc.sector)
    else:
        title = "Lightcurve for target {}".format(lc.label)

    fig = figure(
        title=title,
        height=340,
        width=600,
        tools="pan,wheel_zoom,box_zoom,tap,reset",
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )
    fig.title.offset = -10

    # ylabel: mimic the logic in lc._create_plot()
    ylabel = "Flux"
    if lc.meta.get("NORMALIZED"):
        ylabel = "Normalized " + ylabel
    elif (lc["flux"].unit) and (lc["flux"].unit.to_string() != ""):
        if lc["flux"].unit == (u.electron / u.second):
            yunit_str = "e/s"  # the common case, use abbreviation
        else:
            yunit_str = lc["flux"].unit.to_string()
        ylabel += f" ({yunit_str})"
    fig.yaxis.axis_label = ylabel

    fig.xaxis.axis_label = "Time (days)"
    try:
        if (lc.mission == "K2") or (lc.mission == "Kepler"):
            fig.xaxis.axis_label = "Time - 2454833 (days)"
        elif lc.mission == "TESS":
            fig.xaxis.axis_label = "Time - 2457000 (days)"
    except AttributeError:  # no mission keyword available
        pass

    if ylim_func is None:
        ylims = get_lightcurve_y_limits(lc_source)
    else:
        ylims = _to_unitless(ylim_func(lc))
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add step lines, circles, and hover-over tooltips
    fig.step(
        "time",
        "flux",
        line_width=1,
        color="gray",
        source=lc_source,
        nonselection_line_color="gray",
        nonselection_line_alpha=1.0,
    )
    circ = fig.scatter(
        "time",
        "flux",
        source=lc_source,
        fill_alpha=0.3,
        size=8,
        line_color=None,
        selection_color="firebrick",
        nonselection_fill_alpha=0.0,
        nonselection_fill_color="grey",
        nonselection_line_color=None,
        nonselection_line_alpha=0.0,
        fill_color=None,
        hover_fill_color="firebrick",
        hover_alpha=0.9,
        hover_line_color="white",
    )
    tooltips = [
        ("Cadence", "@cadence"),
        ("Time ({})".format(lc.time.format.upper()), "@time{0,0.000}"),
        ("Time (ISO)", "@time_iso"),
        ("Flux", "@flux"),
        ("Quality Code", "@quality_code"),
        ("Quality Flag", "@quality"),
    ]
    fig.add_tools(
        HoverTool(
            tooltips=tooltips,
            renderers=[circ],
            mode="mouse",
            point_policy="snap_to_data",
        )
    )

    # Vertical line to indicate the cadence
    vertical_line = Span(
        location=lc.time[0].value,
        dimension="height",
        line_color="firebrick",
        line_width=4,
        line_alpha=0.5,
    )
    fig.add_layout(vertical_line)

    return fig, vertical_line


def _add_separation(result: Table, target_coord: SkyCoord):
    """Calculate angular separation of the catalog result from the target."""
    result_coord = SkyCoord(result["RA"], result["DEC"], unit="deg")
    separation = target_coord.separation(result_coord)
    result["Separation"] = separation.arcsec
    result["Separation"].unit = u.arcsec


def add_target_figure_elements(tpf, fig):
    # mark the target's position
    target_ra, target_dec, pm_corrected = _get_corrected_coordinate(tpf)
    target_x, target_y = None, None
    if target_ra is not None and target_dec is not None:
        pix_x, pix_y = tpf.wcs.all_world2pix([(target_ra, target_dec)], 0)[0]
        target_x, target_y = tpf.column + pix_x, tpf.row + pix_y
        fig.scatter(marker="cross", x=target_x, y=target_y, size=20, color="black", line_width=1)
        if not pm_corrected:
            warnings.warn(("Proper motion correction cannot be applied to the target, as none is available. "
                           "Thus the target (the cross) might be noticeably away from its actual position, "
                           "if it has large proper motion."),
                           category=LightkurveWarning)


def init_provider(provider, tpf, magnitude_limit, extra_kwargs=None):
    """Supplying the given provider all the parameters needed (for query, plotting, etc)."""
    if extra_kwargs is None:
        extra_kwargs = {}

    provider_kwargs = extra_kwargs.copy()

    try:
        ra, dec, pm_corrected = _get_corrected_coordinate(tpf)
        c1 = SkyCoord(ra, dec, frame="icrs", unit="deg")
        # OPEN: for now do not issue warning for no PM correction,
        # as users would likely get multiple warnings (from code path of plotting the target as a cross)
    except Exception as err:
        msg = ("Cannot get nearby stars because TargetPixelFile has no valid coordinate. "
               f"ra: {getattr(tpf, 'ra', None)}, dec: {getattr(tpf, 'dec', None)}")
        raise LightkurveError(msg) from err

    provider_kwargs["coord"] = c1

    if provider_kwargs.get("radius") is None:
        # use the default search radius, scaled to the TargePixelFile size

        # Use pixel scale for query size
        pix_scale = 4.0  # arcseconds / pixel for Kepler, default
        if tpf.mission == "TESS":
            pix_scale = 21.0
        # We are querying with a diameter as the radius, overfilling by 2x.
        provider_kwargs["radius"] = Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec")

    provider_kwargs["magnitude_limit"] = magnitude_limit

    provider.init(**provider_kwargs)


def add_catalog_figure_elements(provider, tpf, fig, message_selected_target, arrow_4_selected):

    result = provider.query_catalog()
    # do proper motion correction, if needed
    m = provider.get_proper_motion_correction_meta()
    if m is not None:
        ra_corrected, dec_corrected, _ = _correct_with_proper_motion(
            result[m.ra_colname].quantity, result[m.dec_colname].quantity,
            result[m.pmra_colname].quantity, result[m.pmdec_colname].quantity,
            m.equinox,
            tpf.time[0])
        result["RA"] = ra_corrected.to(u.deg).value
        result["RA"].unit = u.deg
        result["DEC"] = dec_corrected.to(u.deg).value
        result["DEC"].unit = u.deg

    _add_separation(result, provider.coord)

    # Prepare the data source for plotting
    # Convert to pixel coordinates
    result_px_coords = tpf.wcs.all_world2pix(result["RA"], result["DEC"], 0)

    # Gently size the points by their magnitude
    sizes = 64.0 / 2 ** (result["magForSize"] / 5.0)
    source = dict(
        ra=result["RA"],
        dec=result["DEC"],
        x=result_px_coords[0] + tpf.column,
        y=result_px_coords[1] + tpf.row,
        separation=result["Separation"],
        size=sizes,
    )
    provider.add_to_data_source(result, source)
    # Workaround https://github.com/bokeh/bokeh/issues/13904
    # Convert astropy column to plain ndarray, otherwise some columns (MaskedColumn of type int)
    # could raise error during bokeh's serialization:
    # ValueError: cannot convert float NaN to integer
    # ...
    # numpy.ma TypeError: Cannot convert fill_value nan to dtype int64
    for c in source.keys():
        val = source[c]
        if isinstance(val, (MaskedColumn, np.ma.MaskedArray)):
            if np.issubdtype(val.dtype, np.integer):
                # use the default fill_value.
                # It might not be what one actually wants, but that's the best we could do
                val = val.filled()
            else:
                # mimic bokeh's serialization logic (which ignores fill_value, as it's often untrustworthy)
                val = val.filled(np.nan)
        # to be on the safe side, convert astropy Column to nd array to avoid any behavioral difference
        if isinstance(val, Column):
            source[c] = val.value

    source = ColumnDataSource(source)

    #
    # Do the actual UI work
    #

    # 1. plot the data, along with a hover pop-in
    r = fig.scatter(
        "x",
        "y",
        source=source,
        size="size",
        **provider.scatter_kwargs
    )

    fig.add_tools(
        HoverTool(
            tooltips=provider.get_tooltips(),
            renderers=[r],
            mode="mouse",
            point_policy="snap_to_data",
        )
    )

    # 2. render the detail table on click
    def row_to_dict(idx):
        # convert a target at index idx in the column data source to a dict
        return {k: source.data[k][idx] for k in source.data}

    def show_target_info(attr, old, new):
        # the following is essentially redoing the bokeh tooltip template above in plain HTML
        # with some slight tweak, mainly to add some helpful links.
        if len(new) > 0:
            msg = """
<style type="text/css">
    .target_details .error {
        font-size: 80%;
        font-color: gray;
        margin-left: 1ch;
    }
    .target_details .error:before {
        content: "± ";
    }
</style>
Selected:<br>
<table class="target_details">
"""
            for idx in new:
                details, extra_rows = provider.get_detail_view(row_to_dict(idx))
                for header, val_html in details.items():
                    msg += f"<tr><td>{header}</td><td>{val_html}</td>"
                if extra_rows is not None:
                    for row_html in extra_rows:
                        msg += f'<tr><td colspan="2">{row_html}</td></tr>'
                msg += '<tr><td colspan="2">&nbsp;</td></tr>'  # space between multiple targets
            msg += "\n<table>"
            message_selected_target.text = msg
        # else do nothing (not clearing the widget) for now.

    source.selected.on_change("indices", show_target_info)

    # 3. show arrow of the selected target, requires the arrow construct to be supplied to the function
    # display an arrow on the selected target
    def show_arrow_at_target(attr, old, new):
        if len(new) > 0:
            x, y = source.data["x"][new[0]], source.data["y"][new[0]]

            # place the arrow near (x,y), taking care of boundary cases (at the edge of the plot)
            x_midpoint = fig.x_range.start + (fig.x_range.end - fig.x_range.start) / 2
            if x < fig.x_range.start + 1:
                # boundary case: the point is at the left edge of the plot
                arrow_4_selected.x_start = x + 0.85
                arrow_4_selected.x_end = x + 0.2
            elif x > fig.x_range.end - 1:
                # boundary case: the point is at the right edge of the plot
                arrow_4_selected.x_start = x - 0.85
                arrow_4_selected.x_end = x - 0.2
            elif x < x_midpoint:
                # normal case 1 : point is at the left side of the plot
                arrow_4_selected.x_start = x - 0.85
                arrow_4_selected.x_end = x - 0.2
            else:
                # normal case 2 : point is at the right side of the plot
                # flip arrow's direction
                arrow_4_selected.x_start = x + 0.85
                arrow_4_selected.x_end = x + 0.2

            if y > fig.y_range.end - 0.5:
                # boundary case: the point is at near the top of the plot
                arrow_4_selected.y_start = y - 0.4
                arrow_4_selected.y_end = y - 0.1
            elif y < fig.y_range.start + 0.5:
                # boundary case: the point is at near the top of the plot
                arrow_4_selected.y_start = y + 0.4
                arrow_4_selected.y_end = y + 0.1
            else:  # normal case
                arrow_4_selected.y_start = y
                arrow_4_selected.y_end = y

            arrow_4_selected.visible = True
        else:
            arrow_4_selected.visible = False

    source.selected.on_change("indices", show_arrow_at_target)
    return r


def to_selected_pixels_source(tpf_source):
    xx = tpf_source.data["xx"].flatten()
    yy = tpf_source.data["yy"].flatten()
    selected_indices = tpf_source.selected.indices
    return ColumnDataSource(dict(
        xx=xx[selected_indices],
        yy=yy[selected_indices],
    ))


def make_tpf_figure_elements(
    tpf,
    tpf_source,
    tpf_source_selectable=True,
    pedestal=None,
    fiducial_frame=None,
    width=370,
    height=340,
    scale="log",
    vmin=None,
    vmax=None,
    cmap="Viridis256",
    tools="tap,box_select,wheel_zoom,reset",
):
    """Returns the lightcurve figure elements.

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to show.
    tpf_source : bokeh.plotting.ColumnDataSource
        TPF data source.
    tpf_source_selectable : boolean
        True if the tpf_source is selectable. False to show the selected pixels
        in the tpf_source only. Default is True.
    pedestal: float
        A scalar value to be added to the TPF flux values, often to avoid
        taking the log of a negative number in colorbars.
        Defaults to `-min(tpf.flux) + 1`
    fiducial_frame: int
        The tpf slice to start with by default, it is assumed the WCS
        is exact for this frame.
    scale: str
        Color scale for tpf figure. Default is 'log'
    vmin: int [optional]
        Minimum color scale for tpf figure
    vmax: int [optional]
        Maximum color scale for tpf figure
    cmap: str
        Colormap to use for tpf plot. Default is 'Viridis256'
    tools: str
        Bokeh tool list
    Returns
    -------
    fig, stretch_slider : bokeh.plotting.figure.Figure, RangeSlider
    """
    if pedestal is None:
        pedestal = -np.nanmin(tpf.flux.value) + 1
    if scale == "linear":
        pedestal = 0

    if tpf.mission in ["Kepler", "K2"]:
        title = "Pixel data (CCD {}.{})".format(tpf.module, tpf.output)
    elif tpf.mission == "TESS":
        title = "Pixel data (Camera {}.{})".format(tpf.camera, tpf.ccd)
    else:
        title = "Pixel data"

    # We subtract 0.5 from the range below because pixel coordinates refer to
    # the middle of a pixel, e.g. (col, row) = (10.0, 20.0) is a pixel center.
    fig = figure(
        width=width,
        height=height,
        x_range=(tpf.column - 0.5, tpf.column + tpf.shape[2] - 0.5),
        y_range=(tpf.row - 0.5, tpf.row + tpf.shape[1] - 0.5),
        title=title,
        tools=tools,
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )

    fig.yaxis.axis_label = "Pixel Row Number"
    fig.xaxis.axis_label = "Pixel Column Number"

    vlo, lo, hi, vhi = np.nanpercentile(tpf.flux.value + pedestal, [0.2, 1, 95, 99.8])
    if vmin is not None:
        vlo, lo = vmin, vmin
    if vmax is not None:
        vhi, hi = vmax, vmax

    if scale == "log":
        vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    if scale == "linear":
        vstep = (vhi - vlo) / 300.0  # assumes counts >> 1.0!

    if scale == "log":
        color_mapper = LogColorMapper(palette=cmap, low=lo, high=hi)
    elif scale == "linear":
        color_mapper = LinearColorMapper(palette=cmap, low=lo, high=hi)
    else:
        raise ValueError("Please specify either `linear` or `log` scale for color.")

    fig.image(
        [tpf.flux.value[fiducial_frame, :, :] + pedestal],
        x=tpf.column - 0.5,
        y=tpf.row - 0.5,
        dw=tpf.shape[2],
        dh=tpf.shape[1],
        dilate=True,
        color_mapper=color_mapper,
        name="tpfimg",
    )

    # The colorbar will update with the screen stretch slider
    # The colorbar margin increases as the length of the tick labels grows.
    # This colorbar share of the plot window grows, shrinking plot area.
    # This effect is known, some workarounds might work to fix the plot area:
    # https://github.com/bokeh/bokeh/issues/5186

    if scale == "log":
        ticker = LogTicker(desired_num_ticks=8)
    elif scale == "linear":
        ticker = BasicTicker(desired_num_ticks=8)

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=ticker,
        label_standoff=-10,
        border_line_color=None,
        location=(0, 0),
        background_fill_color="whitesmoke",
        major_label_text_align="left",
        major_label_text_baseline="middle",
        title="e/s",
        margin=0,
    )
    fig.add_layout(color_bar, "right")

    color_bar.formatter = PrintfTickFormatter(format="%14i")

    if tpf_source is not None:
        if tpf_source_selectable:
            fig.rect(
                "xx",
                "yy",
                1,
                1,
                source=tpf_source,
                fill_color="gray",
                fill_alpha=0.4,
                line_color="white",
                )
        else:
            # Paint the selected pixels such that they cannot be selected / deselected.
            # Used to show specified aperture pixels without letting users to
            # change them in ``interact_sky```
            selected_pixels_source = to_selected_pixels_source(tpf_source)
            r_selected = fig.rect(
                "xx",
                "yy",
                1,
                1,
                source=selected_pixels_source,
                fill_color="gray",
                fill_alpha=0.0,
                line_color="white",
                )
            r_selected.nonselection_glyph = None

    # Configure the stretch slider and its callback function
    if scale == "log":
        start, end = np.log10(vlo), np.log10(vhi)
        values = (np.log10(lo), np.log10(hi))
    elif scale == "linear":
        start, end = vlo, vhi
        values = (lo, hi)

    stretch_slider = RangeSlider(
        start=start,
        end=end,
        step=vstep,
        title="Screen Stretch ({})".format(scale),
        value=values,
        orientation="horizontal",
        width=200,
        direction="ltr",
        show_value=True,
        sizing_mode="fixed",
        height=15,
        name="tpfstretch",
    )

    def stretch_change_callback_log(attr, old, new):
        """TPF stretch slider callback."""
        fig.select("tpfimg")[0].glyph.color_mapper.high = 10 ** new[1]
        fig.select("tpfimg")[0].glyph.color_mapper.low = 10 ** new[0]

    def stretch_change_callback_linear(attr, old, new):
        """TPF stretch slider callback."""
        fig.select("tpfimg")[0].glyph.color_mapper.high = new[1]
        fig.select("tpfimg")[0].glyph.color_mapper.low = new[0]

    if scale == "log":
        stretch_slider.on_change("value", stretch_change_callback_log)
    if scale == "linear":
        stretch_slider.on_change("value", stretch_change_callback_linear)

    return fig, stretch_slider


def make_default_export_name(tpf, suffix="custom-lc"):
    """makes the default name to save a custom interact mask"""
    fn = tpf.hdu.filename()
    if fn is None:
        outname = "{}_{}_{}.fits".format(tpf.mission, tpf.targetid, suffix)
    else:
        base = os.path.basename(fn)
        outname = base.rsplit(".fits")[0] + "-{}.fits".format(suffix)
    return outname


def show_interact_widget(
    tpf,
    notebook_url=None,
    lc=None,
    max_cadences=200000,
    aperture_mask="default",
    exported_filename=None,
    transform_func=None,
    ylim_func=None,
    vmin=None,
    vmax=None,
    scale="log",
    cmap="Viridis256",
):
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
        For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
        to the public hostname of your JupyterHub and notebook_url will
        be defined appropriately automatically.
    max_cadences: int
        Raise a RuntimeError if the number of cadences shown is larger than
        this value. This limit helps keep browsers from becoming unresponsive.
    aperture_mask : array-like, 'pipeline', 'threshold', 'default', or 'all'
        A boolean array describing the aperture such that `True` means
        that the pixel will be used.
        If None or 'all' are passed, all pixels will be used.
        If 'pipeline' is passed, the mask suggested by the official pipeline
        will be returned.
        If 'threshold' is passed, all pixels brighter than 3-sigma above
        the median flux will be used.
        If 'default' is passed, 'pipeline' mask will be used when available,
        with 'threshold' as the fallback.
    exported_filename: str
        An optional filename to assign to exported fits files containing
        the custom aperture mask generated by clicking on pixels in interact.
        The default adds a suffix '-custom-aperture-mask.fits' to the
        TargetPixelFile basename.
    transform_func: function
        A function that transforms the lightcurve.  The function takes in a
        LightCurve object as input and returns a LightCurve object as output.
        The function can be complex, such as detrending the lightcurve.  In this
        way, the interactive selection of aperture mask can be evaluated after
        inspection of the transformed lightcurve.  The transform_func is applied
        before saving a fits file.  Default: None (no transform is applied).
    ylim_func: function
        A function that returns ylimits (low, high) given a LightCurve object.
        The default is to return an expanded window around the 10-90th
        percentile of lightcurve flux values.
    scale: str
        Color scale for tpf figure. Default is 'log'
    vmin: int [optional]
        Minimum color scale for tpf figure
    vmax: int [optional]
        Maximum color scale for tpf figure
    cmap: str
        Colormap to use for tpf plot. Default is 'Viridis256'
    """
    if _BOKEH_IMPORT_ERROR is not None:
        log.error(
            "The interact() tool requires the `bokeh` Python package; "
            "you can install bokeh using e.g. `conda install bokeh`."
        )
        raise _BOKEH_IMPORT_ERROR

    notebook_url = finalize_notebook_url(notebook_url)

    aperture_mask = tpf._parse_aperture_mask(aperture_mask)
    if ~aperture_mask.any():
        log.error(
            "No pixels in `aperture_mask`, finding optimum aperture using `tpf.create_threshold_mask`."
        )
        aperture_mask = tpf.create_threshold_mask()
    if ~aperture_mask.any():
        log.error("No pixels in `aperture_mask`, using all pixels.")
        aperture_mask = tpf._parse_aperture_mask("all")

    if exported_filename is None:
        exported_filename = make_default_export_name(tpf)
    try:
        exported_filename = str(exported_filename)
    except:
        log.error("Invalid input filename type for interact()")
        raise
    if ".fits" not in exported_filename.lower():
        exported_filename += ".fits"

    if lc is None:
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        tools = "tap,box_select,wheel_zoom,reset"
    else:
        lc = lc.copy()
        tools = "wheel_zoom,reset"
        aperture_mask = np.zeros(tpf.flux.shape[1:]).astype(bool)
        aperture_mask[0, 0] = True

    lc.meta["APERTURE_MASK"] = aperture_mask

    if transform_func is not None:
        lc = transform_func(lc)

    # Bokeh cannot handle many data points
    # https://github.com/bokeh/bokeh/issues/7490
    n_cadences = len(lc.cadenceno)
    if n_cadences > max_cadences:
        log.error(
            f"Error: interact cannot display more than {max_cadences} "
            "cadences without suffering significant performance issues. "
            "You can limit the number of cadences show using slicing, e.g. "
            "`tpf[0:1000].interact()`. Alternatively, you can override "
            "this limitation by passing the `max_cadences` argument."
        )
    elif n_cadences > 30000:
        log.warning(
            f"Warning: the pixel file contains {n_cadences} cadences. "
            "The performance of interact() is very slow for such a "
            "large number of frames. Consider using slicing, e.g. "
            "`tpf[0:1000].interact()`, to make interact run faster."
        )

    def create_interact_ui(doc):
        # The data source includes metadata for hover-over tooltips
        lc_source = prepare_lightcurve_datasource(lc)
        tpf_source = prepare_tpf_datasource(tpf, aperture_mask)

        # Create the lightcurve figure and its vertical marker
        fig_lc, vertical_line = make_lightcurve_figure_elements(
            lc, lc_source, ylim_func=ylim_func
        )

        # Create the TPF figure and its stretch slider
        pedestal = -np.nanmin(tpf.flux.value) + 1
        if scale == "linear":
            pedestal = 0
        fig_tpf, stretch_slider = make_tpf_figure_elements(
            tpf,
            tpf_source,
            pedestal=pedestal,
            fiducial_frame=0,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            cmap=cmap,
            tools=tools,
        )

        # Helper lookup table which maps cadence number onto flux array index.
        tpf_index_lookup = {cad: idx for idx, cad in enumerate(tpf.cadenceno)}

        # Interactive slider widgets and buttons to select the cadence number
        cadence_slider = Slider(
            start=np.min(tpf.cadenceno),
            end=np.max(tpf.cadenceno),
            value=np.min(tpf.cadenceno),
            step=1,
            title="Cadence Number",
            width=490,
        )
        r_button = Button(label=">", button_type="default", width=30)
        l_button = Button(label="<", button_type="default", width=30)
        export_button = Button(
            label="Save Lightcurve", button_type="success", width=120
        )
        message_on_save = Div(text=" ", width=600, height=15)

        # Callbacks
        def _create_lightcurve_from_pixels(
            tpf, selected_pixel_indices, transform_func=transform_func
        ):
            """Create the lightcurve from the selected pixel index list"""
            selected_mask = aperture_mask_from_selected_indices(selected_pixel_indices, tpf)
            lc_new = tpf.to_lightcurve(aperture_mask=selected_mask)
            lc_new.meta["APERTURE_MASK"] = selected_mask
            if transform_func is not None:
                lc_transformed = transform_func(lc_new)
                if len(lc_transformed) != len(lc_new):
                    warnings.warn(
                        "Dropping cadences in `transform_func` is not "
                        "yet supported due to fixed time coordinates."
                        "Skipping the transformation...",
                        LightkurveWarning,
                    )
                else:
                    lc_new = lc_transformed
                    lc_new.meta["APERTURE_MASK"] = selected_mask
            return lc_new

        def update_upon_pixel_selection(attr, old, new):
            """Callback to take action when pixels are selected."""
            # Check if a selection was "re-clicked", then de-select
            if (sorted(old) == sorted(new)) & (new != []):
                # Trigger recursion
                tpf_source.selected.indices = new[1:]

            if new != []:
                lc_new = _create_lightcurve_from_pixels(
                    tpf, new, transform_func=transform_func
                )
                lc_source.data["flux"] = lc_new.flux.value

                if ylim_func is None:
                    ylims = get_lightcurve_y_limits(lc_source)
                else:
                    ylims = _to_unitless(ylim_func(lc_new))
                fig_lc.y_range.start = ylims[0]
                fig_lc.y_range.end = ylims[1]
            else:
                lc_source.data["flux"] = lc.flux.value * 0.0
                fig_lc.y_range.start = -1
                fig_lc.y_range.end = 1

            message_on_save.text = " "
            export_button.button_type = "success"

        def update_upon_cadence_change(attr, old, new):
            """Callback to take action when cadence slider changes"""
            if new in tpf.cadenceno:
                frameno = tpf_index_lookup[new]
                fig_tpf.select("tpfimg")[0].data_source.data["image"] = [
                    tpf.flux.value[frameno, :, :] + pedestal
                ]
                vertical_line.update(location=tpf.time.value[frameno])
            else:
                fig_tpf.select("tpfimg")[0].data_source.data["image"] = [
                    tpf.flux.value[0, :, :] * np.NaN
                ]
            lc_source.selected.indices = []

        def go_right_by_one():
            """Step forward in time by a single cadence"""
            existing_value = cadence_slider.value
            if existing_value < np.max(tpf.cadenceno):
                cadence_slider.value = existing_value + 1

        def go_left_by_one():
            """Step back in time by a single cadence"""
            existing_value = cadence_slider.value
            if existing_value > np.min(tpf.cadenceno):
                cadence_slider.value = existing_value - 1

        def save_lightcurve():
            """Save the lightcurve as a fits file with mask as HDU extension"""
            if tpf_source.selected.indices != []:
                lc_new = _create_lightcurve_from_pixels(
                    tpf, tpf_source.selected.indices, transform_func=transform_func
                )
                lc_new.to_fits(
                    exported_filename,
                    overwrite=True,
                    flux_column_name="SAP_FLUX",
                    aperture_mask=lc_new.meta["APERTURE_MASK"].astype(np.int_),
                    SOURCE="lightkurve interact",
                    NOTE="custom mask",
                    MASKNPIX=np.nansum(lc_new.meta["APERTURE_MASK"]),
                )
                if message_on_save.text == " ":
                    text = '<font color="black"><i>Saved file {} </i></font>'
                    message_on_save.text = text.format(exported_filename)
                    export_button.button_type = "success"
                else:
                    text = '<font color="gray"><i>Saved file {} </i></font>'
                    message_on_save.text = text.format(exported_filename)
            else:
                text = (
                    '<font color="gray"><i>No pixels selected, no mask saved</i></font>'
                )
                export_button.button_type = "warning"
                message_on_save.text = text

        def jump_to_lightcurve_position(attr, old, new):
            if new != []:
                cadence_slider.value = lc.cadenceno[new[0]]

        # Map changes to callbacks
        r_button.on_click(go_right_by_one)
        l_button.on_click(go_left_by_one)
        tpf_source.selected.on_change("indices", update_upon_pixel_selection)
        lc_source.selected.on_change("indices", jump_to_lightcurve_position)
        export_button.on_click(save_lightcurve)
        cadence_slider.on_change("value", update_upon_cadence_change)

        # Layout all of the plots
        sp1, sp2, sp3, sp4 = (
            Spacer(width=15),
            Spacer(width=30),
            Spacer(width=80),
            Spacer(width=60),
        )
        widgets_and_figures = layout(
            [fig_lc, fig_tpf],
            [l_button, sp1, r_button, sp2, cadence_slider, sp3, stretch_slider],
            [export_button, sp4, message_on_save],
        )
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)


def _create_catalog_provider(name):
    if name == "gaiadr3":
        from . import interact_sky_provider_gaia_tic as gaia_t
        return gaia_t.GaiaDR3InteractSkyCatalogProvider()
    elif name == "gaiadr3_tic":
        from . import interact_sky_provider_gaia_tic as gaia_t
        return gaia_t.GaiaDR3TICInteractSkyCatalogProvider()
    elif name == "ztf":
        from . import interact_sky_provider_ztf as ztf
        return ztf.ZTFInteractSkyCatalogProvider()
    elif name == "vsx":
        from . import interact_sky_provider_vsx as vsx
        return vsx.VSXInteractSkyCatalogProvider()
    else:
        raise ValueError(f"Unsupported catalog: {name}")


def show_skyview_widget(tpf, notebook_url=None, aperture_mask="empty", catalogs=None, magnitude_limit=18):
    """skyview

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
        For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
        to the public hostname of your JupyterHub and notebook_url will
        be defined appropriately automatically.
    aperture_mask : array-like, 'pipeline', 'threshold', 'default', 'background', or 'empty'
        Highlight pixels selected by aperture_mask.
        Default is 'empty': no pixel is highlighted.
    magnitude_limit : float
        A value to limit the results in based on Gaia Gmag. Default, 18.
    """
    if _BOKEH_IMPORT_ERROR is not None:
        log.error(
            "The interact_sky() tool requires the `bokeh` Python package; "
            "you can install bokeh using e.g. `conda install bokeh`."
        )
        raise _BOKEH_IMPORT_ERROR

    notebook_url = finalize_notebook_url(notebook_url)

    if catalogs is None:
        if tpf.mission == "TESS":
            catalogs = ["gaiadr3_tic"]
        else:
            catalogs = ["gaiadr3"]

    # Try to identify the "fiducial frame", for which the TPF WCS is exact
    zp = (tpf.pos_corr1 == 0) & (tpf.pos_corr2 == 0)
    (zp_loc,) = np.where(zp)

    if len(zp_loc) == 1:
        fiducial_frame = zp_loc[0]
    else:
        fiducial_frame = 0

    aperture_mask = tpf._parse_aperture_mask(aperture_mask)

    def create_interact_ui(doc):
        tpf_source = prepare_tpf_datasource(tpf, aperture_mask)

        # The data source includes metadata for hover-over tooltips

        # Create the TPF figure and its stretch slider
        fig_tpf, stretch_slider = make_tpf_figure_elements(
            tpf,
            tpf_source,
            tpf_source_selectable=False,
            fiducial_frame=fiducial_frame,
            width=640,
            height=600,
            tools="tap,box_zoom,wheel_zoom,reset"
        )

        # Add a marker (cross) to indicate the coordinate of the target
        add_target_figure_elements(tpf, fig_tpf)

        # a widget that displays some of the selected star's metadata
        # so that they can be copied (e.g., GAIA ID).
        # Generally the content is a more detailed version of the on-hover tooltip.
        message_selected_target = Div(text="")

        # an arrow that serves as the marker of the selected star.
        arrow_4_selected = Arrow(
            end=VeeHead(size=16, fill_color="red", line_color="black"),
            line_color="red", line_width=4,
            x_start=0, y_start=0, x_end=0, y_end=0, tags=["selected"],
            visible=False,
        )
        fig_tpf.add_layout(arrow_4_selected)

        providers = []
        catalog_renderers = []
        for catalog_spec in catalogs:
            if isinstance(catalog_spec, (tuple, list)):
                provider, extra_kwargs = catalog_spec
            else:
                provider, extra_kwargs = catalog_spec, None

            if isinstance(provider, str):
                provider = _create_catalog_provider(provider)
            # else assume it's a InteractSkyCatalogProvider object

            # pass all the parameters for query, plotting, etc. to the provider
            init_provider(provider, tpf, magnitude_limit, extra_kwargs=extra_kwargs)
            renderer = add_catalog_figure_elements(provider, tpf, fig_tpf, message_selected_target, arrow_4_selected)
            providers.append(provider)
            catalog_renderers.append(renderer)

        # Optionally override the default title
        if tpf.mission == "K2":
            fig_tpf.title.text = (
                "Skyview for EPIC {}, K2 Campaign {}, CCD {}.{}".format(
                    tpf.targetid, tpf.campaign, tpf.module, tpf.output
                )
            )
        elif tpf.mission == "Kepler":
            fig_tpf.title.text = (
                "Skyview for KIC {}, Kepler Quarter {}, CCD {}.{}".format(
                    tpf.targetid, tpf.quarter, tpf.module, tpf.output
                )
            )
        elif tpf.mission == "TESS":
            fig_tpf.title.text = "Skyview for TESS {} Sector {}, Camera {}.{}".format(
                tpf.targetid, tpf.sector, tpf.camera, tpf.ccd
            )

        # Layout all of the plots
        if len(catalogs) < 2:
            widgets_and_figures = layout([fig_tpf, message_selected_target], stretch_slider)
        else:
            select_catalog_ui = CheckboxGroup(
                labels=[p.label for p in providers],
                active=list(range(0, len(catalogs))),  # make all checked
                inline=True,
                )
            # add more horizontal spacing between checkboxes
            select_catalog_ui.stylesheets = [InlineStyleSheet(css="""\
.bk-input-group.bk-inline > label {
    margin: 5px 10px;
}
""")]

            def select_catalog_handler(attr, old, new):
                # new is the list of indices of active (i.e., checked) catalogs
                for i in range(len(catalog_renderers)):
                    r = catalog_renderers[i]
                    if i in new:
                        r.visible = True
                    else:
                        r.visible = False

            select_catalog_ui.on_change("active", select_catalog_handler)
            widgets_and_figures = layout([fig_tpf, message_selected_target], select_catalog_ui, stretch_slider)
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
